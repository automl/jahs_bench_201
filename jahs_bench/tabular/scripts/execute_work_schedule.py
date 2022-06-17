"""
This script is intended to be used for resuming halted evaluations after the data from a partial run has been used to
pre-compute a better task distribution across multiple workers on a cluster.
"""


import argparse
import ConfigSpace
import logging
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import traceback
from typing import Tuple, Sequence

import jahs_bench.tabular.lib.naslib.utils.logging as naslib_logging
import jahs_bench.tabular.lib.naslib.utils.utils as naslib_utils
from jahs_bench.tabular.lib.naslib.utils.utils import AttrDict

from jahs_bench.tabular.clusterlib import prescheduler as sched_utils
from jahs_bench.tabular.lib.core.constants import Datasets, standard_task_metrics, training_config
from jahs_bench.tabular.lib.core import utils
from jahs_bench.tabular.lib.core import datasets as dataset_lib
from jahs_bench.tabular.lib.core.procs import train
from jahs_bench.tabular.search_space import NASB201HPOSearchSpace


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rootdir", type=Path, default=Path().cwd(),
                        help="Path to the root directory where all the tasks' output will be stored. Task-specific "
                             "sub-directories will be created here if needed. Note that this is NOT the same as the "
                             "'basedir' of a DirectoryTree object as multiple DirectoryTree objects will be created "
                             "as and when needed.")
    parser.add_argument("--workerid", type=int,
                        help="An offset from 0 for this worker within the current job's allocation of workers. This "
                             "does NOT correspond to a single 'taskid' but rather the portfolio of configurations that "
                             "this worker will evaluate.")
    parser.add_argument("--workerid_offset", type=int, default=0,
                        help="An additional fixed offset from 0 for this worker that, combined with the value of "
                             "'--worker', gives the overall worker ID in the context of a job that has been split into "
                             "multiple smaller parts.")
    parser.add_argument("--datadir", type=Path, default=dataset_lib.get_default_datadir(),
                        help="The directory where all datasets are expected to be stored.")
    parser.add_argument("--portfolio", type=Path, default=None,
                        help="Path to a pickled Pandas Series that contains the portfolio for this job's workers.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")

    # Unpack the training config overrides into CLI flags.
    # for k, v in recognized_train_config_overrides.items():
    for k, v in training_config.items():
        parser.add_argument(f"--{k}", **v)

    return parser


def parse_training_overrides(args: argparse.Namespace) -> AttrDict:
    return AttrDict({k: getattr(args, k) for k in recognized_train_config_overrides.keys()})


def get_tranining_config_from_args(args: Sequence[str]) -> AttrDict:
    return AttrDict({k: getattr(args, k) for k in training_config.keys()})


def reload_train_config(dtree: utils.DirectoryTree, **overrides) -> Tuple[AttrDict, Datasets]:
    """ Load a saved training config from disk, overriding the training config with appropriate values if needed. """

    with open(dtree.task_config_file, "r") as fp:
        task_config = json.load(fp)

    task_config["train_config"] = {**task_config["train_config"],
                                   **{k: v for k, v in overrides.items() if v is not None}}

    try:
        dataset = [v for v in Datasets.__members__.values() if v.value[0] == task_config["dataset"]][0]
    except IndexError:
        raise RuntimeError(f"Unable to determine appropriate dataset with name - {task_config['dataset']}.")

    # Update the saved task config with the latest overrides.
    ## DISABLED since this can lead to race conditions during multiprocess write operations
    # with open(dtree.task_config_file, "w") as fp:
    #     json.dump(task_config, fp, default=str)

    return AttrDict(task_config["train_config"]), dataset


def load_model_config(task_metrics: AttrDict, model_idx: int) -> Tuple[int, dict]:
    """ Given the metrics of a task, recover the specified model's configuration. Remember that 'model_idx' uses a
    starting index of 1 and not 0. """

    # n_known_samples = len(task_metrics.model_idx)
    # if n_known_samples < model_idx:
    try:
        global_seed = task_metrics.global_seed[model_idx - 1]
        model_config = task_metrics.model_config[model_idx - 1]
    except IndexError as e:
        raise IndexError(f"The specified model index {model_idx} was not found in the task's recorded metrics, which "
                         f"contained {len(task_metrics.model_idx)} records.") from e

    return global_seed, model_config


def instantiate_model(model_config: dict, dataset: Datasets) -> NASB201HPOSearchSpace:
    """ Given a model configuration dict and a dataset, instantiate a model with the corresponding configuration. """

    NASB201HPOSearchSpace.NUM_CLASSES = dataset.value[3]
    search_space = NASB201HPOSearchSpace()
    if dataset.value[2] == 1:  # Check # of channels
        # The dataset is in Grayscale
        search_space.GRAYSCALE = True

    assert utils.adapt_search_space(search_space, portfolio=None, taskid=None, opts=model_config, suffix=None), \
        f"Unable to restrict the search space to use the specified model configuration:\n{model_config}"
    assert all([isinstance(h, ConfigSpace.Constant) for h in search_space.config_space.get_hyperparameters()]), \
        f"Failed to restrict all hyperparameters in the search space to the specified model configuration. The " \
        f"restricted space is defined by:\n{search_space.config_space}"

    # Since all hyper-parameters are constants, this will always sample the same configuration.
    search_space.sample_random_architecture()

    return search_space

# TODO: Extend this script to also include stage 2 verification functionality - loading a specific checkpoint, training
#  it for a specified number of epochs, and verifying the metric data. Save the new metrics as alternative data points
#  for a potential analysis of the various metric distributions
def resume_work(basedir: Path, taskid: int, model_idx: int, datadir: Path, debug: bool = False,
                logger: logging.Logger = None, **training_config_overrides):
    """ Resume working on a particular configuration wherever it was left off by loading all relevant parameters from
    the saved checkpoints. Some training parameters may be overridden by providing the relevant overrides as keyword
    arguments. """

    dir_tree = utils.DirectoryTree(basedir=basedir, taskid=taskid, model_idx=model_idx)
    if logger is None:
        logger = naslib_logging.setup_logger(str(dir_tree.model_dir / f"resume.log"))

    task_metrics = AttrDict(utils.attrdict_factory(metrics=standard_task_metrics, template=list))
    # No timer, since this is read-only - reads the pre-sampled configs from disk
    _ = utils.MetricLogger(dir_tree=dir_tree, metrics=task_metrics, timer=None,
                           set_type=utils.MetricLogger.MetricSet.task, logger=logger)

    try:
        global_seed, model_config = load_model_config(task_metrics, model_idx)
    except IndexError:
        logger.warning(f"Could not resume working on model config of task {taskid}, model index {model_idx} due to an "
                       f"error:", exc_info=True)
        return

    try:
        train_config, dataset = reload_train_config(dir_tree, **training_config_overrides)
    except RuntimeError:
        logger.warning(f"Could not resume working on model config of task {taskid}, model index {model_idx} due to an "
                       f"error:", exc_info=True)
        return

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    ## Model initialization
    model = instantiate_model(model_config, dataset)
    model.parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Training model {model_idx} on device {str(device)}")

    transfer_devices = device.type != "cpu"
    if transfer_devices:
        model = model.to(device)

    try:
        naslib_utils.set_seed(global_seed)

        data_loaders, min_shape = dataset_lib.get_dataloaders(
            dataset=dataset, batch_size=train_config.batch_size, cutout=0, split=train_config.split,
            resolution=model_config.get("Resolution", 0), trivial_augment=model_config.get("TrivialAugment", False),
            datadir=datadir
        )
        validate = "valid" in data_loaders
        train(
            model=model, data_loaders=data_loaders, train_config=train_config, dir_tree=dir_tree, logger=logger,
            validate=validate, transfer_devices=transfer_devices, device=device, debug=debug, min_shape=min_shape
        )

    except Exception as e:
        logger.info("Architecture Training failed.", exc_info=True)
        error_description = {
            "exception": traceback.format_exc(),
            "config": model_config,
            "global_seed": global_seed
        }
        with open(dir_tree.model_error_description_file, "w") as fp:
            json.dump(error_description, fp, indent=4)
        if debug:
            raise e
    else:
        if dir_tree.model_error_description_file.exists():
            dir_tree.model_error_description_file.unlink()
        del model  # Release memory
        dir_tree.model_idx = None  # Ensure that previously written data cannot be overwritten

    ## Clean up memory before terminating task
    del task_metrics
    del dir_tree


if __name__ == "__main__":
    ## Parse CLI
    args = argument_parser().parse_args()

    train_config_overrides = get_tranining_config_from_args(args)

    workerid = args.workerid + args.workerid_offset

    logdir: Path = args.rootdir / sched_utils.logdir_name
    logdir.mkdir(exist_ok=True, parents=False)
    logger = naslib_logging.setup_logger(str(logdir / f"{workerid}.log"), name='tabular_sampling')
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    portfolio = pd.read_pickle(args.portfolio)
    if "worker_id" not in portfolio.index.names:
        raise ValueError(f"The given portfolio at {args.portfolio} has not been configured properly, its index lacks "
                         f"the level for a worker ID. Portfolio index level names: {portfolio.index.names}")
    if workerid not in portfolio.index.unique("worker_id"):
        logger.warning(f"Worker {workerid} not present in the job's portfolio. Ignore this warning if the job "
                       f"specification had more workers than needed, otherwise, check the portfolio file "
                       f"{args.portfolio}")
        sys.exit(1)

    portfolio = portfolio.xs(workerid, level="worker_id")
    # TODO: Remove worker-specific portfolio_dir and relative functionality once it is confirmed to be superfluous
    worker_config = sched_utils.WorkerConfig(worker_id=workerid, portfolio=portfolio)

    for taskid, model_idx, basedir in worker_config.iterate_portfolio(rootdir=args.rootdir):
        resume_work(basedir, taskid, model_idx, datadir=args.datadir, debug=args.debug, logger=logger,
                    **train_config_overrides)

