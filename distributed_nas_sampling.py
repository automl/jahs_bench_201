import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Iterable, Callable, Dict, List

import ConfigSpace
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from naslib.search_spaces import NASB201HPOSearchSpace
import naslib.utils.utils as naslib_utils
from naslib.utils.utils import AverageMeter, AttrDict
import naslib.utils.logging as naslib_logging
from naslib.tabular_sampling.utils import utils
from naslib.tabular_sampling.utils.custom_nasb201_code import CosineAnnealingLR

# Randomly generated entropy source, to remain fixed across experiments.
seed = 79029434164686768057103648623012072794

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, default=Path().cwd(),
                        help="Path to the base directory where all the tasks' output will be stored. Task-specific "
                             "sub-directories will be created here if needed.")
    parser.add_argument("--taskid", type=int,
                        help="An offset from 0 for this task within the current node's allocation of tasks.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of epochs that each sampled architecture should be trained for. Default: 25")
    parser.add_argument("--resize", type=int, default=0,
                        help="An integer value (8, 16, 32, ...) to determine the scaling of input images. Default: 0 - "
                             "don't use resize.")
    parser.add_argument("--global-seed", type=int, default=None,
                        help="A value for a global seed to be used for all global NumPy and PyTorch random operations. "
                             "This is different from the fixed seed used for reproducible search space sampling. If not "
                             "specified, a random source of entropy is used instead.")
    parser.add_argument("--standalone-mode", action="store_true",
                        help="Switch that enables working in a single-task local setup as opposed to the default "
                             "multi-node cluster setup.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    parser.add_argument("--use_grad_clipping", action="store_true", help="Enable gradient clipping for SGD.")
    parser.add_argument("--split", action="store_true", help="Split training dataset into training and validation "
                                                             "sets.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of samples per mini-batch.")
    # parser.add_argument("--cutout", type=float, default=0.,
    #                     help="Cutout probability for applying stochastic cutout to training data during pre-processing. "
    #                          "Settings this to 0. disables cutout (default). [DISABLED]")
    parser.add_argument("--warmup_epochs", type=int, default=0.,
                        help="When set to a positive integer, this many epochs are used to warm-start the training.")
    parser.add_argument("--generate_sampling_profile", action="store_true",
                        help="When given, does not actually train the sampled models. Instead, only samples are "
                             "repeatedly drawn at 1s intervals and saved in order to build a profile of expected "
                             "samples.")
    parser.add_argument("--nsamples", type=int, default=100,
                        help="Only used when --generate_sampling_profile is given. Sets the number of samples to be "
                             "drawn from the search space.")
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


def _get_common_metrics(extra_metrics: Optional[List[str]] = None) -> AttrDict:
    metrics = AttrDict(
        duration = AverageMeter(),
        forward_duration = AverageMeter(),
        data_load_duration = AverageMeter(),
        loss = AverageMeter(),
        acc = AverageMeter(),
        **({m: AverageMeter() for m in extra_metrics} if extra_metrics else {})
    )
    return metrics


def _main_proc(model, dataloader: Iterable, loss_fn: Callable, optimizer: torch.optim.Optimizer,
               scheduler: CosineAnnealingLR, mode: str, device: str, epoch_metrics: Dict[str, List],
               debug: bool = False, use_grad_clipping: bool = True,
               summary_writer: torch.utils.tensorboard.SummaryWriter = None, name: Optional[str] = None, logger = None):

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

    metrics = _get_common_metrics(extra_metrics=extra_metrics)

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
            lrs.append(scheduler.get_lr()[0]) # Assume that there is only one parameter group

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

    n = 0 # Number of individual data points processed
    for key, value in metrics.items():
        epoch_metrics[key].append(value.avg) # Metrics are recorded as averages per data-point for each epoch
        n = max([n, value.cnt]) # value.cnt is expected to be constant

    return n, False


def train(model: NASB201HPOSearchSpace, data_loaders, train_config: AttrDict, logger, validate=False):
    """
    Train the given model using the given data loaders and training configuration. Returns a dict containing various metrics.

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
    start_time, latency = time.time(), AverageMeter()
    train_duration, eval_duration = AverageMeter(), AverageMeter() # Useful for internal diagnostics only

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Training model with config: {model.config} on device {device}")
    model.parse()
    transfer_devices = device.type != "cpu"
    if transfer_devices:
        model = model.to(device)

    train_queue = data_loaders["train"]
    test_queue = data_loaders["test"]
    if validate:
        valid_queue = data_loaders["valid"]

    extra_metrics = ["data_transfer_duration"] if transfer_devices else []
    train_metrics = AttrDict(
        {k: [] for k in _get_common_metrics(extra_metrics=extra_metrics + ["backprop_duration"]).keys()})
    valid_metrics = AttrDict({k: [] for k in _get_common_metrics(extra_metrics=extra_metrics).keys()})
    test_metrics = AttrDict({k: [] for k in _get_common_metrics(extra_metrics=extra_metrics).keys()})

    optimizer, scheduler, loss_fn = construct_model_optimizer(model, train_config)
    logger.debug(f"Initialized optimizer: {optimizer.__class__.__name__}")

    if debug:
        summary_writer = SummaryWriter(model_tensorboard_dir)
        inp_size = train_config.batch_size, 3, train_config.resize or 32, train_config.resize or 32
        model.eval()
        with torch.no_grad():
            summary_writer.add_graph(model, torch.tensor(np.random.random(inp_size).astype(np.float32)).to(device))
    else:
        summary_writer = None

    train_size = valid_size = test_size = 0

    for e in range(train_config.epochs):
        ## Handle training set
        dataloader = train_queue
        epoch_metrics = train_metrics
        n, diverged = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                 scheduler=scheduler, mode="train", device=device, epoch_metrics=epoch_metrics,
                                 use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                                 name="Training", logger=logger, debug=debug)
        if diverged:
            break
        train_size = max([train_size, n])
        train_duration.update(train_metrics.duration[-1]) # We don't need to assign a weight to this value

        if validate:
            ## Handle validation set
            dataloader = valid_queue
            epoch_metrics = valid_metrics
            with torch.no_grad():
                n, _ = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                  scheduler=scheduler, mode="eval", device=device, epoch_metrics=epoch_metrics,
                                  use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                                  name="Validation", logger=logger, debug=debug)
            valid_size = max([valid_size, n])
            latency.update(valid_metrics.forward_duration[-1], valid_size)
            eval_duration.update(valid_metrics.duration[-1], valid_size)
            # Note: eval_duration DOES require a weight - it mixes validation and test sets if validate=True

        ## Handle test set
        dataloader = test_queue
        epoch_metrics = test_metrics
        with torch.no_grad():
            n, _ = _main_proc(model=model, dataloader=dataloader, loss_fn=loss_fn, optimizer=optimizer,
                              scheduler=scheduler, mode="eval", device=device, epoch_metrics=epoch_metrics,
                              use_grad_clipping=train_config.use_grad_clipping, summary_writer=summary_writer,
                              name="Test", logger=logger, debug=debug)
        test_size = max([test_size, n])
        latency.update(test_metrics.forward_duration[-1], test_size)
        eval_duration.update(test_metrics.duration[-1], test_size)

        ## Logging
        naslib_logging.log_every_n_seconds(
            logging.DEBUG,
            f"[Epoch {e + 1}/{train_config.epochs}] " +
            f"Avg. Train Loss: {train_metrics.loss[-1]:.5f}, Avg. Train Acc: {train_metrics.acc[-1]:.5f},\n" +
            (f"Avg. Valid Loss: {valid_metrics.loss[-1]}, Avg. Valid Acc: {valid_metrics.acc[-1]},\n" if validate else "") +
            f"Avg. Test Loss: {test_metrics.loss[-1]}\, Avg. Test Acc: {test_metrics.acc[-1]}\n" +
            f"Time Elapsed: {time.time() - start_time}",
            15,
            name=logger.name
        )

    end_time = time.time()

    if debug:
        tb_metrics = AttrDict(
            train_loss=train_metrics.loss,
            train_acc=train_metrics.acc,
            test_loss=test_metrics.loss,
            test_acc=test_metrics.acc,
            **(dict(valid_loss=valid_metrics.loss, valid_acc=valid_metrics.acc,) if validate else {})
        )
        # If training diverged, the last epoch was not completed and should be looked at in detail instead.
        num_epochs = e - 1 if diverged else e
        if num_epochs >= 0:
            for i in range(num_epochs):
                summary_writer.add_scalars("metrics", {k: v[i] for k, v in tb_metrics.items()})

        summary_writer.flush()
        summary_writer.close()

    # Prepare metrics
    raw_metrics = AttrDict()
    record_metrics = [train_metrics, valid_metrics, test_metrics] if validate else [train_metrics, test_metrics]
    metric_types = ["train", "valid", "test"] if validate else ["train", "test"]
    for k1, metrics in zip(metric_types, record_metrics):
        for k2, data in metrics.items():
            raw_metrics[(k1, k2)] = data

    if diverged:
        raise RuntimeError(f"Model training has diverged after {e} epochs.")

    job_metrics = AttrDict()
    job_metrics.latency = latency.avg
    job_metrics.avg_train_duration = train_duration.avg
    job_metrics.avg_eval_duration = eval_duration.avg
    job_metrics.runtime = end_time - start_time

    return raw_metrics, job_metrics


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

    init_time = time.time()

    ## Parse CLI
    args = argument_parser().parse_args()
    basedir = args.basedir
    global_seed = args.global_seed

    # Pseudo-RNG should rely on a bit-stream that is largely uncorrelated both within and across tasks
    rng = np.random.RandomState(np.random.Philox(seed=seed, counter=args.taskid))

    taskdir: Path = basedir / str(args.taskid)
    outdir: Path = taskdir / "benchmark_data"
    outdir.mkdir(exist_ok=True, parents=True)

    if global_seed is None:
        global_seed = int(np.random.default_rng().integers(0, 2 ** 32 - 1))

    naslib_utils.set_seed(global_seed)

    logger = naslib_logging.setup_logger(str(taskdir.resolve() / "log.log"))
    with open(taskdir / "training_config.json", "w") as fp:
        json.dump(vars(args), fp, default=str)

    if args.debug:
        tensorboard_logdir = taskdir / "tensorboard_logs"
        model_tensorboard_dir = None  # To be initialized later
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    naslib_utils.log_args(vars(args))

    search_space = NASB201HPOSearchSpace()
    adapt_search_space(search_space, args.opts)

    # TODO: Verify if removing this is fine
    # data_loaders_start_wctime = time.time()
    # data_loaders_start_ptime = time.process_time()
    #
    # data_loaders_end_wctime = time.time()
    # data_loaders_end_ptime = time.process_time()
    #
    # init_duration = data_loaders_start_wctime - init_time
    # data_loaders_wc_duration = data_loaders_end_wctime - data_loaders_start_wctime
    # data_loaders_proc_duration = data_loaders_end_ptime - data_loaders_start_ptime
    #
    # with open(outdir / "meta.json", "w") as fp:
    #     json.dump(dict(
    #         init_duration=init_duration,
    #         wc_duration=data_loaders_wc_duration,
    #         proc_duration=data_loaders_proc_duration,
    #         resize=args.resize,
    #         epochs=args.epochs
    #     ), fp, indent=4)

    n_archs = 0

    if args.generate_sampling_profile:
        from signal import signal, SIGINT

        sampled_configs = []

        def write_sampling_profile():
            with open(outdir / "sample_profile.json", "w") as fp:
                json.dump(sampled_configs, fp)

        def handle_sigint_while_sampling(signal_received, frame):
            write_sampling_profile()
            sys.exit(0)

        signal(SIGINT, handle_sigint_while_sampling)

    while True:
        model: NASB201HPOSearchSpace = search_space.clone()
        model.sample_random_architecture(rng=rng)
        model_config = model.config.get_dictionary()
        naslib_logging.log_every_n_seconds(
            logging.INFO,
            f"Sampled new architecture: {model_config} from space {search_space.__class__.__name__}",
            15,
            name=logger.name
        )

        if args.debug:
            model_tensorboard_dir = tensorboard_logdir / str(n_archs)

        if args.generate_sampling_profile:
            n_archs += 1
            sampled_configs.append(model_config)
            time.sleep(1)

            if n_archs >= args.nsamples:
                write_sampling_profile()
                break
            else:
                continue

        try:
            n_archs += 1

            data_loaders, _, _ = utils.get_dataloaders(dataset="cifar10", batch_size=args.batch_size, cutout=0,
                                                       split=args.split, resize=model_config.get("resolution", 0))
            validate = "valid" in data_loaders
            raw_metrics, job_metrics = train(model=model, data_loaders=data_loaders, train_config=AttrDict(vars(args)),
                                             logger=logger, validate=validate)
        except Exception as e:
            naslib_logging.log_every_n_seconds(logging.INFO, "Architecture Training failed.", 15, name=logger.name)
            job_metrics = {"exception": str(e)}
            if args.debug:
                raise e
        else:
            ## Clean-up after nominal program execution
            naslib_logging.log_every_n_seconds(logging.INFO, "Finished training architecture.", 15, name=logger.name)
            del(model) # Release memory
            metric_df = pd.DataFrame()
            for k, v in raw_metrics.items():
                metric_df[k] = v

            for k, v in model_config.items():
                metric_df[("config", k)] = v

            metric_df.assign(**job_metrics)
            metric_df.index = metric_df.index.set_names(["Epoch"]) + 1
            metric_df.to_pickle(outdir / f"{n_archs}.pkl") # Don't compress yet

            max_idx = metric_df.index.max()
            job_metrics["final_train_acc"] = metric_df[("train", "acc")][max_idx]
            if validate:
                job_metrics["final_val_acc"] = metric_df[("valid", "acc")][max_idx]
            job_metrics["final_test_acc"] = metric_df[("test", "acc")][max_idx]

            del(metric_df) # Release memory
        finally:
            job_metrics["config"] = model_config
            with open(outdir / f"{n_archs}.json", "w") as fp:
                json.dump(job_metrics, fp, indent=4)
