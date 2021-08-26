import argparse
import codecs
import json
import logging
import os
import sys
import time
from pathlib import Path

import ConfigSpace
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from naslib.search_spaces import NASB201HPOSearchSpace
import naslib.utils.utils as naslib_utils
from naslib.utils.utils import AverageMeter
import naslib.utils.logging as naslib_logging
from naslib.tabular_sampling.custom_nasb201_code import CosineAnnealingLR, CrossEntropyLabelSmooth

init_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=Path, default=Path().cwd(),
                    help="Path to the base directory where all the tasks' output will be stored. Task-specific "
                         "sub-directories will be created here if needed.")
parser.add_argument("--nb201_data", type=Path, default=Path().cwd() / "nasb201_full_pandas.pkl.gz",
                    help="Path to a pickled pandas DataFrame, containing the relevant data from NASBench-201 as a "
                         "pandas DataFrame.")
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
parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamW", "SGD"], default="SGD",
                    help="Which optimizer to use for training the models.")

parser.add_argument("--use_grad_clipping", action="store_true", help="Enable gradient clipping for SGD.")
args = parser.parse_args()

basedir = args.basedir
taskid = args.taskid
epochs = args.epochs
resize = args.resize
global_seed = args.global_seed
optimizer_choice = args.optimizer
nb201_data: pd.DataFrame = pd.read_pickle(args.nb201_data)
# The top100 configs from the original NASBench-201 dataset, sorted from worst to best as per evaluation accuracy
candidates = nb201_data.index.tolist()

taskdir: Path = basedir / str(taskid)
outdir: Path = taskdir / "benchmark_data"
outdir.mkdir(exist_ok=True, parents=True)

# Randomly generated entropy source, to remain fixed across experiments.
seed = 79029434164686768057103648623012072794
# Pseudo-RNG should rely on a bit-stream that is largely uncorrelated both within and across tasks
rng = np.random.RandomState(np.random.Philox(seed=seed, counter=taskid))

if global_seed is None:
    global_seed = int(np.random.default_rng().integers(0, 2**32 - 1))

# Read args and config, setup logger
if torch.cuda.is_available():
    naslib_args = [f"--gpu={torch.cuda.device_count() - 1}"]
else:
    naslib_args = []

naslib_args += [
        # "config_file": "%s/defaults/nas_sampling.yaml" % (naslib_utils.get_project_root()),
        f"--seed={str(global_seed)}",
        f"--resize={str(resize)}",
        "out_dir", str(taskdir),
        "search.epochs", str(epochs),
        "search.train_portion", str(0.5),
        "search.batch_size", str(256),
]
naslib_args = naslib_utils.default_argument_parser().parse_args(naslib_args)

naslib_config = naslib_utils.get_config_from_args(naslib_args, config_type="nas")
naslib_utils.set_seed(naslib_config.seed)

logger = naslib_logging.setup_logger(naslib_config.save + "/log.log")
if args.debug:
    tensorboard_logdir = Path(naslib_config.save) / "tensorboard_logs"
    model_tensorboard_dir = None    # To be initialized later
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)
naslib_utils.log_args(naslib_config)

def init_adam(model):
    config = model.config
    lr, weight_decay = config["learning_rate"], config["weight_decay"]
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.debug("Initialized optimizer: Adam")
    return optim

def init_adamw(model):
    config = model.config
    lr, weight_decay = config["learning_rate"], config["weight_decay"]
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    logger.debug("Initialized optimizer: AdamW")
    return optim

def init_sgd(model):
    config = model.config
    lr, momentum, weight_decay = config["learning_rate"], config["momentum"], config["weight_decay"]
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    logger.debug("Initialized optimizer: SGD")
    return optim

optimizer_constructor = {
    "Adam": init_adam,
    "AdamW": init_adamw,
    "SGD": init_sgd,
}


def _test_proc(model, loss_fn, test_queue, device, metrics, errors_dict):
    model.eval()
    metrics.test_loss.reset()
    metrics.test_acc.reset()
    test_start_time = time.time()
    with torch.no_grad():
        test_data_load_start = time.time()
        for (test_inputs, test_labels) in test_queue:
            metrics.test_data_load_time.update(time.time() - test_data_load_start)
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            latency_start_time = time.time()
            logits_test = model(test_inputs)
            metrics.latency.update(time.time() - latency_start_time)
            test_loss = loss_fn(logits_test, test_labels)
            metrics.test_loss.update(float(test_loss.detach().cpu()))
            update_accuracies(metrics, logits_test, test_labels, "test")

            test_data_load_start = time.time()
    test_duration = time.time() - test_start_time

    errors_dict.test_acc.append(metrics.test_acc.avg)
    errors_dict.test_loss.append(metrics.test_loss.avg)
    errors_dict.test_time.append(test_duration)
    errors_dict.test_data_load_time.append(metrics.test_data_load_time.avg)


def _main_proc(model, dataloader, optimizer, scheduler, mode, device, epoch_metrics, use_grad_clipping=True):
    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unrecognized mode '{mode}'.")

    ## Setup control flags
    transfer_devices = device != "cpu"
    train_model = mode == "train"

    ## Setup metrics
    if transfer_devices:
        data_transfer_time = AverageMeter()

    if train_model:
        backprop_duration = AverageMeter()

    forward_duration = AverageMeter()
    data_load_duration = AverageMeter()
    step_duration = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    ## Iterate over mini-batches
    data_load_start_time= time.time()
    for step, (inputs, labels) in enumerate(dataloader):
        start_time, duration = time.time(), AverageMeter()
        data_load_duration.update(start_time - data_load_start_time)
        scheduler.update(None, step)

        if train_model:
            optimizer.zero_grad()

        if transfer_devices:
            inputs = inputs.to(device)
            labels = labels.to(device)
            data_transfer_time.update(time.time() - start_time)

        ## Forward Pass
        forward_start_time = time.time()
        logits = model(inputs)
        loss = loss(logits, labels)
        forward_duration.update(time.time() - forward_start_time)

        ## Backward Pass
        if train_model:
            backprop_start_time = time.time()
            loss.backward()
            if use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            end_time = time.time()
            backprop_duration.update(end_time - backprop_start_time)
        else:
            end_time = time.time()

        ## Bookkeeping
        step_duration.update(end_time - start_time)
        losses.update(loss.detach().cpu())
        acc = naslib_utils.accuracy(logits.detach().cpu(), labels.detach().cpu(), topk=(1,))
        accuracies.update(acc.data.item())





def train(model, data_loaders, train_config):
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

    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transfer_devices = device != "cpu"
    logger.debug(f"Training model with config: {model.config} on device {device}")
    model.parse()
    model = model.to(device)

    errors_dict, metrics = get_metrics(model)
    train_queue, valid_queue, test_queue, _, _ = data_loaders

    optim = optimizer_constructor[model.config["optimizer"]](model)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_config.epochs, eta_min=0.)
    scheduler = CosineAnnealingLR(optim, warmup_epochs=0, epochs=train_config.epochs, T_max=train_config.epochs,
                                  eta_min=0.)
    loss = torch.nn.CrossEntropyLoss()
    # loss = CrossEntropyLabelSmooth()

    if args.debug:
        summary_writer = SummaryWriter(model_tensorboard_dir)
        model.eval()
        inp_size = naslib_config.search.batch_size, 3, naslib_config.resize or 32, naslib_config.resize or 32
        with torch.no_grad():
            summary_writer.add_graph(model, torch.tensor(np.random.random(inp_size).astype(np.float32)).to(device))

    diverged = False
    for e in range(train_config.epochs):
        metrics.train_acc.reset()
        metrics.train_loss.reset()
        metrics.val_acc.reset()
        metrics.val_loss.reset()
        metrics.epoch_train_time.reset()
        metrics.train_data_transfer_time.reset()
        metrics.train_backprop_time.reset()

        # Update the current epoch in the scheduler
        scheduler.update(e, None)

        epoch_errors_dict = naslib_utils.AttrDict(
            {'train_acc': [],
             'train_loss': [],
             'val_acc': [],
             'val_loss': [],
             'grad_norms': []
             })

        train_data_load_start = time.time()
        for step, ((train_inputs, train_labels), (val_inputs, val_labels)) in \
                enumerate(zip(train_queue, valid_queue)):

            # Update the current minibatch iteration in the scheduler for the ongoing epoch
            scheduler.update(None, float(step))

            metrics.train_data_load_time.update(time.time() - train_data_load_start)
            ## Training procedure
            train_start_time = time.time()
            model.train()
            if transfer_devices:
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)
                metrics.train_data_transfer_time.update(time.time() - train_start_time)
            optim.zero_grad()
            logits_train = model(train_inputs)
            train_loss = loss(logits_train, train_labels)
            train_backprop_start_time = time.time()
            train_loss.backward()
            if optimizer_choice == "SGD" and args.use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optim.step()
            train_end_time = time.time()
            metrics.train_backprop_time.update(train_end_time - train_backprop_start_time)
            metrics.epoch_train_time.update(train_end_time - train_start_time)

            ## Validation procedure
            model.eval()
            with torch.no_grad():
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                logits_val = model(val_inputs)
                val_loss = loss(logits_val, val_labels)

            ## Logging, debug
            if torch.isnan(train_loss):
                logger.debug(f"Epoch {e}-{step} - Training Predictions: {logits_train}\n\nExpected labels: "
                             f"{train_labels}\n\nLoss: {train_loss} ")
                diverged = True

            if torch.isnan(val_loss):
                logger.debug(f"Epoch {e}-{step} - Validation Predictions: {logits_val}\n\nExpected labels: "
                             f"{val_labels}\n\nLoss: {val_loss} ")
                diverged = True

            naslib_logging.log_every_n_seconds(
                logging.DEBUG,
                "Epoch {}-{}, Train loss: {:.5f}, validation loss: {:.5f}".format(e, step, train_loss, val_loss),
                n=15, name=logger.name
            )

            metrics.train_loss.update(float(train_loss.detach().cpu()))
            metrics.val_loss.update(float(val_loss.detach().cpu()))
            update_accuracies(metrics, logits_train, train_labels, "train")
            update_accuracies(metrics, logits_val, val_labels, "val")

            if args.debug:
                epoch_errors_dict.train_loss.append(metrics.train_loss.avg)
                epoch_errors_dict.train_acc.append(metrics.train_acc.avg)
                epoch_errors_dict.val_loss.append(metrics.val_loss.avg)
                epoch_errors_dict.val_acc.append(metrics.val_acc.avg)
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in model.parameters()]),
                    2).detach().cpu()
                epoch_errors_dict.grad_norms.append(grad_norm)

            if diverged:
                if args.debug:
                    for i in range(len(epoch_errors_dict.train_loss)):
                        summary_writer.add_scalars(
                            "divergent_epoch_metrics", {k: v[i] for k, v in epoch_errors_dict.items()}, i)
                break

            train_data_load_start = time.time()

        # scheduler.step()

        errors_dict.train_acc.append(metrics.train_acc.avg)
        errors_dict.train_loss.append(metrics.train_loss.avg)
        errors_dict.val_loss.append(metrics.val_loss.avg)
        errors_dict.val_acc.append(metrics.val_acc.avg)
        errors_dict.train_time.append(metrics.epoch_train_time.sum)
        errors_dict.train_data_load_time.append(metrics.train_data_load_time.avg)
        if transfer_devices:
            errors_dict.train_data_transfer_time.append(metrics.train_data_transfer_time.sum)
        errors_dict.train_backprop_time.append(metrics.train_backprop_time.sum)

        _test_proc(model, loss, test_queue, device, metrics, errors_dict)

        if diverged:
            break

    errors_dict.runtime = time.time() - start_time
    errors_dict.latency = metrics.latency.avg / train_config.batch_size

    if args.debug:
        for i in range(len(errors_dict.train_loss)):
            summary_writer.add_scalars("metrics", {k: v[i] for k, v in errors_dict.items() if k in
                                                   ["train_acc", "train_loss", "val_acc", "val_loss"]}, i)

    return errors_dict


def get_metrics(model):
    errors_dict = naslib_utils.AttrDict(
        {'train_acc': [],
         'train_loss': [],
         'val_acc': [],
         'val_loss': [],
         'test_acc': [],
         'test_loss': [],
         'runtime': None,
         'train_time': [],
         'test_time': [],
         'train_data_load_time': [],
         'train_data_transfer_time': [],
         'train_backprop_time': [],
         'test_data_load_time': [],
         'model_size_MB': naslib_utils.count_parameters_in_MB(model)}
    )

    metrics = naslib_utils.AttrDict({
        'train_acc': naslib_utils.AverageMeter(),
        'train_loss': naslib_utils.AverageMeter(),
        'val_acc': naslib_utils.AverageMeter(),
        'val_loss': naslib_utils.AverageMeter(),
        'test_acc': naslib_utils.AverageMeter(),
        'test_loss': naslib_utils.AverageMeter(),
        'epoch_train_time': naslib_utils.AverageMeter(),
        'latency': naslib_utils.AverageMeter(),
        'train_data_load_time': naslib_utils.AverageMeter(),
        'train_data_transfer_time': naslib_utils.AverageMeter(),
        'train_backprop_time': naslib_utils.AverageMeter(),
        'test_data_load_time': naslib_utils.AverageMeter(),
    })

    return errors_dict, metrics


def update_accuracies(metrics, logits, target, split):
    """Update the accuracy counters"""
    logits = logits.clone().detach().cpu()
    target = target.clone().detach().cpu()
    acc, _ = naslib_utils.accuracy(logits, target, topk=(1, 5))
    n = logits.size(0)

    if split == 'train':
        metrics.train_acc.update(acc.data.item(), n)
    elif split == 'val':
        metrics.val_acc.update(acc.data.item(), n)
    elif split == 'test':
        metrics.test_acc.update(acc.data.item(), n)
    else:
        raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")


search_space = NASB201HPOSearchSpace()

data_loaders_start_wctime = time.time()
data_loaders_start_ptime = time.process_time()

data_loaders = naslib_utils.get_train_val_loaders(naslib_config, mode='train')

data_loaders_end_wctime = time.time()
data_loaders_end_ptime = time.process_time()

init_duration = data_loaders_start_wctime - init_time
data_loaders_wc_duration = data_loaders_end_wctime - data_loaders_start_wctime
data_loaders_proc_duration = data_loaders_end_ptime - data_loaders_start_ptime

with open(outdir / "meta.json", "w") as fp:
    json.dump(dict(
        init_duration=init_duration,
        wc_duration=data_loaders_wc_duration,
        proc_duration=data_loaders_proc_duration,
        resize=resize,
        epochs=epochs
    ), fp, indent=4)

n_archs = 0

config = {
    "N": 5,
    "W": 3,
    "optimizer": optimizer_choice,
    "learning_rate": 0.1,
    "weight_decay": 5e-4,
}

if optimizer_choice == "SGD":
    config["momentum"] = 0.9

def sample_architecture():
    global config
    choice = candidates[rng.randint(len(candidates))]
    for i in range(1, 7):
        config[f"Op{i}"] = choice[i-1]


if __name__ == "__main__":
    while True:
        naslib_logging.log_every_n_seconds(logging.DEBUG, f"Sampling architecture #{n_archs + 1}.", 15, name=logger.name)
        model: NASB201HPOSearchSpace = search_space.clone()
        sample_architecture()
        model.config = ConfigSpace.Configuration(model.config_space, config)
        model.clear()
        model._construct_graph()

        if args.debug:
            model_tensorboard_dir = tensorboard_logdir / str(n_archs)

        try:
            res = train(model=model, data_loaders=data_loaders, train_config=naslib_config.search)
        except Exception as e:
            res = {"exception": str(e)}
            if args.debug:
                raise e # TODO: Remove for production runs
        res["config"] = model.config.get_dictionary()
        naslib_logging.log_every_n_seconds(logging.DEBUG, "Finished training architecture.", 15, name=logger.name)
        n_archs += 1
        with open(outdir / f"{n_archs}.json", "w") as fp:
            json.dump(res, fp, indent=4)
