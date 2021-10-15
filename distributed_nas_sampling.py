"""
Main script for sampling from a specified joint NAS+HPO search space, training and evaluating the sampled
configurations and recording the metrics when run on a server. Intended for use with a SLURM script such that one job
launches multiple parallel python processes (tasks), each running any specified number of logical CPUs. Each task then
evaluates a number of configurations (models). The data is generated and stored using the following directory structure
template:
<base_dir>    <-- Possibly a job-level directory
|    |--> <task_idx>/
|    |    |--> metrics/
|    |    |    |--> <task-level metrics, if any>
|    |    |--> models /
|    |    |--> <model_idx>/
|    |    |    |--> metrics/
|    |    |    |    |--> <model-level metrics, if any>
|    |    |    |--> checkpoints/
|    |    |    |    |--> <model checkpoints, if any>

The PRNGs are setup in such a way that, for the hard-coded, fixed seed value, the same task, as identified by an
integer task_id, will generate the same sequence of samples from any given search space as well as the same sequence of
global seeds for numpy/torch/random.

"""

import argparse
import json
import logging
import time
from itertools import repeat
from pathlib import Path
from typing import Optional, Iterable, Callable, Dict, List

import ConfigSpace
import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

import naslib.utils.logging as naslib_logging
import naslib.utils.utils as naslib_utils
from naslib.search_spaces import NASB201HPOSearchSpace
from naslib.tabular_sampling.utils import utils
from naslib.tabular_sampling.utils.custom_nasb201_code import CosineAnnealingLR
from naslib.utils.utils import AverageMeter, AttrDict
from naslib.tabular_sampling.utils.count_flops import get_model_flops

# Randomly generated entropy source, to remain fixed across experiments.
seed = 79029434164686768057103648623012072794


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, default=Path().cwd(),
                        help="Path to the base directory where all the tasks' output will be stored. Task-specific "
                             "sub-directories will be created here if needed.")
    parser.add_argument("--taskid", type=int,
                        help="An offset from 0 for this task within the current node's allocation of tasks.")
    parser.add_argument("--taskid_base", type=int, default=0,
                        help="An additional offset from 0 to manually shift all task IDs further, useful when sampling "
                             "is intended to continue adding more data beyond a previous job's data.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs that each sampled architecture should be trained for. Default: 25")
    parser.add_argument("--global_seed", type=int, default=None,
                        help="A value for a global seed to be used for all global NumPy and PyTorch random "
                             "operations. This is different from the fixed seed used for reproducible search space "
                             "sampling. If not specified, a random source of entropy is used instead.")
    parser.add_argument("--standalone-mode", action="store_true",
                        help="Switch that enables working in a single-task local setup as opposed to the default "
                             "multi-node cluster setup.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    parser.add_argument("--use_grad_clipping", action="store_true", help="Enable gradient clipping for SGD.")
    parser.add_argument("--split", action="store_true",
                        help="Split training dataset into training and validation sets.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of samples per mini-batch.")
    parser.add_argument("--warmup_epochs", type=int, default=0.,
                        help="When set to a positive integer, this many epochs are used to warm-start the training.")
    parser.add_argument("--generate_sampling_profile", action="store_true",
                        help="When given, does not actually train the sampled models. Instead, only samples are "
                             "repeatedly drawn at 1s intervals and saved in order to build a profile of expected "
                             "samples.")
    parser.add_argument("--nsamples", type=int, default=100,
                        help="Only used when --generate_sampling_profile is given. Sets the number of samples to be "
                             "drawn from the search space.")
    parser.add_argument("--disable_checkpointing", action="store_true",
                        help="When given, checkpointing of model training is disabled. By default, model training is "
                             "checkpointed either every X seconds or Y epochs, whichever occurs first. Check "
                             "--checkpoint_interval_seconds and --checkpoint_interval_epochs.")
    parser.add_argument("--checkpoint_interval_seconds", type=int, default=1800,
                        help="The time interval between subsequent model training checkpoints, in seconds. Default: "
                             "30 minutes i.e. 1800 seconds.")
    parser.add_argument("--checkpoint_interval_epochs", type=int, default=20,
                        help="The interval between subsequent model training checkpoints, in epochs. Default: "
                             "20 epochs.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="A variable number of optional keyword arguments provided as 2-tuples, each potentially "
                             "corresponding to a hyper-parameter in the search space. If a match is found, that "
                             "hyper-parameter is excluded from the search space and fixed to the given value instead.")
    return parser


def construct_model_optimizer(model: NASB201HPOSearchSpace, train_config: AttrDict):
    optim_type = utils.optimizers.__members__[model.config["optimizer"]]
    optimizer = optim_type.construct(model, model.config)
    if optim_type == utils.optimizers.SGD:
        # Potentially stabilize SGD with Warm-up
        scheduler = CosineAnnealingLR(optimizer, warmup_epochs=train_config.warmup_epochs, epochs=train_config.epochs,
                                      T_max=train_config.epochs, eta_min=0.)
    else:
        # No need to use warmup epochs for Adam and AdamW
        scheduler = CosineAnnealingLR(optimizer, warmup_epochs=0, epochs=train_config.epochs,
                                      T_max=train_config.epochs, eta_min=0.)
    loss_fn = torch.nn.CrossEntropyLoss()

    return optimizer, scheduler, loss_fn


def _main_proc(model, dataloader: Iterable, loss_fn: Callable, optimizer: torch.optim.Optimizer,
               scheduler: CosineAnnealingLR, mode: str, device: str, epoch_metrics: Dict[str, List],
               debug: bool = False, use_grad_clipping: bool = True,
               summary_writer: torch.utils.tensorboard.SummaryWriter = None, name: Optional[str] = None, logger=None):
    # Logging setup
    name_str = f"Dataset: {name} " if name else ""
    if not logger:
        logger = logging.getLogger(__name__)

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unrecognized mode '{mode}'.")

    ## Setup control flags
    transfer_devices = device.type != "cpu"
    train_model = mode == "train"

    ## Setup metrics
    extra_metrics = []
    # metrics = _get_common_metrics()

    if transfer_devices:
        extra_metrics.append("data_transfer_duration")
        # metrics["data_transfer_time"] = AverageMeter()

    if train_model:
        extra_metrics.append("backprop_duration")
        # metrics["backprop_duration"] = AverageMeter()

    metrics = utils.get_common_metrics(extra_metrics=extra_metrics)

    if debug:
        losses = []
        accs = []
        grad_norms = []
        lrs = []
        assert isinstance(summary_writer, torch.utils.tensorboard.SummaryWriter), \
            "When using debug mode, a tensorboard summary writer object must be passed."

    ## Iterate over mini-batches
    diverged = False
    data_load_start_time = time.time()
    nsteps = len(dataloader)
    for step, (inputs, labels) in enumerate(dataloader):
        metric_weight = inputs.size(0)
        start_time = time.time()
        metrics.data_load_duration.update(start_time - data_load_start_time, metric_weight)

        if train_model:
            scheduler.update(None, 1. * step / nsteps)
            optimizer.zero_grad()

        if transfer_devices:
            inputs = inputs.to(device)
            labels = labels.to(device)
            metrics.data_transfer_duration.update(time.time() - start_time, metric_weight)

        ## Forward Pass
        forward_start_time = time.time()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        metrics.forward_duration.update(time.time() - forward_start_time, metric_weight)

        ## Backward Pass
        if train_model:
            backprop_start_time = time.time()
            loss.backward()
            if use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            end_time = time.time()
            metrics.backprop_duration.update(end_time - backprop_start_time, metric_weight)
        else:
            end_time = time.time()

        ## Bookkeeping
        metrics.duration.update(end_time - start_time, metric_weight)
        metrics.loss.update(loss.detach().cpu().data.item(), metric_weight)
        acc = naslib_utils.accuracy(logits.detach().cpu(), labels.detach().cpu(), topk=(1,))[0]
        metrics.acc.update(acc.data.item(), metric_weight)

        ## Debugging, logging and clean-up
        naslib_logging.log_every_n_seconds(
            logging.DEBUG,
            f"[Minibatch {step + 1}/{nsteps}] {name_str}Mode: {mode} Avg. Loss: {metrics.loss.avg:.5f}, "
            f"Avg. Accuracy: {metrics.acc.avg:.5f}, Avg. Time per step: {metrics.duration.avg}",
            n=30, name=logger.name
        )

        if debug:
            losses.append(loss.detach().cpu())
            accs.append(acc.data.item())
            grad_norms.append(torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in model.parameters()]),
                2).detach().cpu())
            lrs.append(scheduler.get_lr()[0])  # Assume that there is only one parameter group

        if torch.isnan(loss):
            logger.debug(f"Encountered NAN loss. Predictions: {logits}\n\nExpected labels: {labels}\n\nLoss: {loss}")
            diverged = True

        if diverged:
            if debug:
                divergent_metrics = {
                    "Minibatch Losses": losses,
                    "Minibatch Accuracies": accs,
                    "Minibatch LRs": lrs,
                    "Minibatch Gradient Norms": grad_norms
                }
                for i in range(step):
                    summary_writer.add_scalars(
                        "divergent_epoch_metrics", {k: v[i] for k, v in divergent_metrics.items()}, i)
                summary_writer.flush()
            return 0, True

        data_load_start_time = time.time()

    n = 0  # Number of individual data points processed
    for key, value in metrics.items():
        epoch_metrics[key].append(value.avg)  # Metrics are recorded as averages per data-point for each epoch
        n = max([n, value.cnt])  # value.cnt is expected to be constant

    return n, False


def train(model: NASB201HPOSearchSpace, data_loaders, train_config: AttrDict, dir_tree: utils.DirectoryTree,
          logger: logging.Logger, validate=False, transfer_devices: bool = False,
          device: [torch.DeviceObjType, str] = "cpu"):
    """
    Train the given model using the given data loaders and training configuration. Returns a dict containing
    various metrics.

    Parameters
    ----------
    model: naslib.search_spaces.Graph
        A NASLib object obtained by sampling a random architecture on a search space. Ideally, the sampled object should
        be cloned before being passsed. The model will be parsed to PyTorch before training.
    data_loaders: Tuple
        A tuple of objects that correspond to the data loaders generated by naslib.utils.utils.get_train_val_loaders().
    train_config: naslib.utils.utils.AttrDict
        An attribute dict containing various configuration parameters for the model training.
    """

    debug = train_config.debug
    start_time = time.time()
    latency = AverageMeter()

    # Initialization calls to psutil
    _ = psutil.cpu_percent()
    _ = psutil.virtual_memory()
    _ = psutil.swap_memory()

    train_queue = data_loaders["train"]
    test_queue = data_loaders["test"]
    if validate:
        valid_queue = data_loaders["valid"]

    extra_metrics = ["data_transfer_duration"] if transfer_devices else []
    model_metrics = AttrDict({
        "train": utils.get_common_metrics(extra_metrics=extra_metrics + ["backprop_duration"], template=list),
        "valid": utils.get_common_metrics(extra_metrics=extra_metrics, template=list),
        "test": utils.get_common_metrics(extra_metrics=extra_metrics, template=list),
        "diagnostic": AttrDict({k: list() for k in ["latency", "runtime", "cpu_percent", "memory_ram", "memory_swap"]}),
    })

    optimizer, scheduler, loss_fn = construct_model_optimizer(model, train_config)
    logger.debug(f"Initialized optimizer: {optimizer.__class__.__name__}")

    ## Initialize checkpoint function and metric logger, load existing checkpoints and metrics
    old_chkpt_runtime = 0.
    old_chkpt_epochs = -1
    if not train_config.disable_checkpointing:
        checkpoint = utils.Checkpointer(
            model=model, optimizer=optimizer, scheduler=scheduler,
            interval_seconds=train_config.checkpoint_interval_seconds,
            interval_epochs=train_config.checkpoint_interval_epochs, dir_tree=dir_tree,
            logger=logger, map_location=device)
        old_chkpt_runtime = checkpoint.runtime
        old_chkpt_epochs = checkpoint.elapsed_epochs

    model_metrics_logger = utils.MetricLogger(
        dir_tree=dir_tree, metrics=model_metrics,
        log_interval=None if train_config.disable_checkpointing else train_config.checkpoint_interval_seconds,
        set_type=utils.MetricLogger.MetricSet.model, logger=logger
    )

    if debug:
        summary_writer = SummaryWriter(model_tensorboard_dir)
        inp_size = train_config.batch_size, 3, model.config["resolution"], model.config["resolution"] or 32
        model.eval()
        with torch.no_grad():
            summary_writer.add_graph(model, torch.tensor(np.random.random(inp_size).astype(np.float32)).to(device))
    else:
        summary_writer = None

    train_size = valid_size = test_size = 0

    diverged = False

    for e in range(old_chkpt_epochs + 1, train_config.epochs):
        ## Handle training set
        dataloader = train_queue
        epoch_metrics = model_metrics.train
        n, diverged = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                 scheduler=scheduler, mode="train", device=device, epoch_metrics=epoch_metrics,
                                 use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                                 name="Training", logger=logger, debug=debug)
        if diverged:
            break
        train_size = max([train_size, n])

        ## Handle validation set, if needed
        if validate:
            dataloader = valid_queue
            epoch_metrics = model_metrics.valid
            with torch.no_grad():
                n, _ = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                  scheduler=scheduler, mode="eval", device=device, epoch_metrics=epoch_metrics,
                                  use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                                  name="Validation", logger=logger, debug=debug)
            valid_size = max([valid_size, n])
            latency.update(model_metrics.valid.forward_duration[-1], valid_size)
            # Note: eval_duration DOES require a weight - it mixes validation and test sets if validate=True

        ## Handle test set
        dataloader = test_queue
        epoch_metrics = model_metrics.test
        with torch.no_grad():
            n, _ = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                              scheduler=scheduler, mode="eval", device=device, epoch_metrics=epoch_metrics,
                              use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                              name="Test", logger=logger, debug=debug)
        test_size = max([test_size, n])
        latency.update(model_metrics.test.forward_duration[-1], test_size)

        ## Logging
        naslib_logging.log_every_n_seconds(
            logging.DEBUG,
            f"[Epoch {e + 1}/{train_config.epochs}] " +
            f"Avg. Train Loss: {model_metrics.train.loss[-1]:.5f}, " +
            f"Avg. Train Acc: {model_metrics.train.acc[-1]:.5f},\n" +
            (f"Avg. Valid Loss: {model_metrics.valid.loss[-1]:.5f}, " +
             f"Avg. Valid Acc: {model_metrics.valid.acc[-1]:.5f},\n" if validate else "") +
            f"Avg. Test Loss: {model_metrics.test.loss[-1]:.5f}, " +
            f"Avg. Test Acc: {model_metrics.test.acc[-1]:.5f}\n" +
            f"Time Elapsed: {time.time() - start_time}",
            15,
            name=logger.name
        )

        ## Checkpointing
        # Add a one-time offset to the runtime in case an old checkpoint was loaded
        effective_elapsed_runtime = time.time() - start_time + old_chkpt_runtime
        last_epoch = e == train_config.epochs - 1
        if not train_config.disable_checkpointing:
            checkpoint(
                runtime=effective_elapsed_runtime,
                elapsed_epochs=e,
                force_checkpoint=last_epoch
            )

        model_metrics.diagnostic.latency.append(latency.avg)
        model_metrics.diagnostic.runtime.append(effective_elapsed_runtime)
        model_metrics.diagnostic.cpu_percent.append(psutil.cpu_percent())
        model_metrics.diagnostic.memory_ram.append(psutil.virtual_memory().available)
        model_metrics.diagnostic.memory_swap.append(psutil.swap_memory().free)
        model_metrics_logger.log(elapsed_runtime=effective_elapsed_runtime, force=last_epoch)

    if debug:
        tb_metrics = AttrDict(
            train_loss=model_metrics.train.loss,
            train_acc=model_metrics.train.acc,
            test_loss=model_metrics.test.loss,
            test_acc=model_metrics.test.acc,
            **(dict(valid_loss=model_metrics.valid.loss, valid_acc=model_metrics.valid.acc, ) if validate else {})
        )
        # If training diverged, the last epoch was not completed and should be looked at in detail instead.
        num_epochs = train_config.epochs if not diverged else e - 1
        if num_epochs >= 0:
            for i in range(num_epochs):
                summary_writer.add_scalars("metrics", {k: v[i] for k, v in tb_metrics.items()})

        summary_writer.flush()
        summary_writer.close()

    if diverged:
        raise RuntimeError(f"Model training has diverged after {e} epochs.")

    if not train_config.disable_checkpointing:
        del checkpoint
    del model_metrics_logger

    return model_metrics


def sample_architecture(config, candidates, rng):
    choice = candidates[rng.randint(len(candidates))]
    for i in range(1, 7):
        config[f"Op{i}"] = choice[i - 1]
    return config


def adapt_search_space(original_space: NASB201HPOSearchSpace, opts):
    if opts is None:
        return
    config_space = original_space.config_space
    known_params = {p.name: p for p in config_space.get_hyperparameters()}

    def param_interpretor(param, value):
        known_config_space_value_types = {
            ConfigSpace.UniformIntegerHyperparameter: int,
            ConfigSpace.UniformFloatHyperparameter: float,
            ConfigSpace.CategoricalHyperparameter: lambda x: type(param.choices[0])(x),
            ConfigSpace.OrdinalHyperparameter: lambda x: type(param.sequence[0])(x),
        }
        return known_config_space_value_types[type(param)](value)

    i = iter(opts)
    for arg, val in zip(i, i):
        if arg in known_params:
            old_param = known_params[arg]
            known_params[arg] = ConfigSpace.Constant(arg, param_interpretor(old_param, val),
                                                     meta=dict(old_param.meta, **dict(constant_overwrite=True)))
    new_config_space = ConfigSpace.ConfigurationSpace(f"{config_space.name}_custom")
    new_config_space.add_hyperparameters(known_params.values())

    original_space.config_space = new_config_space
    logger.info(f"Modified original search space's hyperparameter config space using constant values. "
                f"New config space: \n{original_space.config_space}")


if __name__ == "__main__":

    ## Parse CLI
    args = argument_parser().parse_args()
    basedir = args.basedir
    global_seed = args.global_seed

    # Pseudo-RNG should rely on a bit-stream that is largely uncorrelated both within and across tasks
    real_taskid = args.taskid_base + args.taskid
    rng = np.random.RandomState(np.random.Philox(seed=seed, counter=real_taskid))

    dir_tree = utils.DirectoryTree(basedir=basedir, taskid=real_taskid)

    if global_seed is None:
        def seeds():
            while True:
                yield rng.randint(0, 2 ** 32 - 1)

        global_seed = seeds()
    else:
        global_seed = repeat(global_seed)

    logger = naslib_logging.setup_logger(str(dir_tree.task_dir / "log.log"))
    task_metrics = AttrDict({
        "model_idx": [],
        "model_config": [],
        "global_seed": [],
        "size_MB": [],
        "FLOPS": [],
    })
    task_metric_logger = utils.MetricLogger(dir_tree=dir_tree, metrics=task_metrics, log_interval=None,
                                            set_type=utils.MetricLogger.MetricSet.task, logger=logger)
    n_known_samples = len(task_metrics.model_idx)

    with open(dir_tree.task_dir / "task_config.json", "w") as fp:
        json.dump(vars(args), fp, default=str)

    if args.debug:
        tensorboard_logdir = dir_tree.task_dir / "tensorboard_logs"
        model_tensorboard_dir = None  # To be initialized later
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    naslib_utils.log_args(vars(args))

    search_space = NASB201HPOSearchSpace()
    adapt_search_space(search_space, args.opts)

    def sampler():
        while True:
            curr_global_seed = next(global_seed)
            naslib_utils.set_seed(curr_global_seed)
            model: NASB201HPOSearchSpace = search_space.clone()
            model.sample_random_architecture(rng=rng)
            model_config = model.config.get_dictionary()
            yield model, model_config, curr_global_seed

    if args.generate_sampling_profile:
        sampled_configs = []

    train_config = AttrDict(vars(args))

    for model_idx, (model, model_config, curr_global_seed) in enumerate(sampler(), start=1):
        logger.info(f"Sampled new architecture: {model_config} from space {search_space.__class__.__name__}")

        ## Model initialization
        model.parse()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Training model {model_idx} on device {str(device)}")
        transfer_devices = device.type != "cpu"
        if transfer_devices:
            model = model.to(device)

        ## Handle task-level metrics
        if model_idx <= n_known_samples:
            # This particular model has already been sampled once. Verify config and seed.
            assert task_metrics.global_seed[model_idx - 1] == curr_global_seed, \
                f"There is a mismatch between the previously registered global seed used for evaluating the " \
                f"model index {model_idx} and the newly generated seed {curr_global_seed}"
            assert task_metrics.model_config[model_idx - 1] == model_config, \
                f"There is a mismatch between the previously registered model config used for evaluating the " \
                f"model index {model_idx} and the newly generated config {model_config}"
        else:
            # A new model config has been sampled.
            task_metrics.model_idx.append(model_idx)
            task_metrics.model_config.append(model_config)
            task_metrics.global_seed.append(curr_global_seed)
            task_metrics.size_MB.append(naslib_utils.count_parameters_in_MB(model))
            task_metrics.FLOPS.append(get_model_flops(
                model=model,
                input_shape=(train_config.batch_size, 3, model_config["resolution"], model_config["resolution"]),
                transfer_devices=transfer_devices,
                device=device
            ))
            task_metric_logger.log(elapsed_runtime=model_idx, force=True)

        if args.debug:
            model_tensorboard_dir = tensorboard_logdir / str(model_idx)

        if args.generate_sampling_profile:
            time.sleep(1)

            if model_idx >= args.nsamples:
                break
            else:
                continue

        ## Actual model training and evaluation
        try:
            naslib_utils.set_seed(curr_global_seed)
            data_loaders, _, _ = utils.get_dataloaders(dataset="cifar10", batch_size=args.batch_size, cutout=0,
                                                       split=args.split, resize=model_config.get("resolution", 0))
            validate = "valid" in data_loaders
            dir_tree.model_idx = model_idx
            train(
                model=model, data_loaders=data_loaders,
                train_config=train_config, dir_tree=dir_tree,
                logger=logger, validate=validate,
                transfer_devices=transfer_devices, device=device
            )
        except Exception as e:
            naslib_logging.log_every_n_seconds(logging.INFO, "Architecture Training failed.", 15, name=logger.name)
            error_description = {
                "exception": str(e),
                "config": model_config,
                "global_seed": curr_global_seed
            }
            with open(dir_tree.model_dir / f"error_description.json", "w") as fp:
                json.dump(error_description, fp, indent=4)
            if args.debug:
                raise e
        else:
            ## Clean-up after nominal program execution
            logger.info("Sampled architecture trained successfully.")

            del model  # Release memory
            dir_tree.model_idx = None  # Ensure that previously written data cannot be overwritten

    ## Clean up memory before terminating task
    del task_metrics
    del task_metric_logger
    del dir_tree
