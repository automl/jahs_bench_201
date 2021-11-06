import logging
import time
from typing import Optional, Iterable, Callable, Dict, List

import numpy as np
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter

import naslib.utils.logging as naslib_logging
import naslib.utils.utils as naslib_utils
from naslib.search_spaces import NASB201HPOSearchSpace
from naslib.tabular_sampling.lib import utils
from naslib.tabular_sampling.lib.custom_nasb201_code import CosineAnnealingLR
from naslib.utils.utils import AverageMeter, AttrDict


def construct_model_optimizer(model: NASB201HPOSearchSpace, train_config: AttrDict):
    optim_type = utils.optimizers.__members__[model.config["Optimizer"]]
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
        # TODO: Remove swap memory metric, there is no swap memory on the cluster anyways
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
        summary_writer = SummaryWriter(dir_tree.model_tensorboard_dir)
        inp_size = train_config.batch_size, 3, model.config["Resolution"] or 32, model.config["Resolution"] or 32
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
