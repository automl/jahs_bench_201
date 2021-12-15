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
import torch
import traceback
from typing import Tuple, Sequence

import naslib.utils.logging as naslib_logging
import naslib.utils.utils as naslib_utils
from naslib.utils.utils import AttrDict

from tabular_sampling.clusterlib import prescheduler as sched_utils
from tabular_sampling.lib.core.constants import Datasets, standard_task_metrics, training_config
from tabular_sampling.lib.core import datasets as dataset_lib, utils
from tabular_sampling.lib.core.procs import train
from tabular_sampling.search_space import NASB201HPOSearchSpace

# _log = logging.getLogger(__name__)
# recognized_train_config_overrides = {
#     "epochs": dict(
#         type=int, default=None,
#         help="Number of epochs that each sampled architecture should be trained for. If not specified, uses the value "
#              "specified in the original training configuration."),
#     "disable_checkpointing": dict(
#         action="store_true",
#         help="When given, checkpointing of model training is disabled. By default, model training is checkpointed "
#              "either every X seconds or Y epochs, whichever occurs first. Check --checkpoint_interval_seconds and "
#              "--checkpoint_interval_epochs."),
#     "checkpoint_interval_seconds": dict(
#         type=int, default=None,
#         help="The time interval between subsequent model training checkpoints, in seconds. If not specified, uses the "
#              "value specified in the original training configuration."),
#     "checkpoint_interval_epochs": dict(
#         type=int, default=None,
#         help="The interval between subsequent model training checkpoints, in epochs. If not specified, uses the value "
#              "specified in the original training configuration.")
# }


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
    parser.add_argument("--portfolio_dir", type=Path, default=None,
                        help="Path to a directory from where each worker will be able to read its own allocated "
                             "portfolio of configurations to evaluate.")
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

    n_known_samples = len(task_metrics.model_idx)
    if n_known_samples < model_idx:
        raise IndexError(f"The specified model index {model_idx} was not found in the task's recorded metrics.")

    global_seed = task_metrics.global_seed[model_idx - 1]
    model_config = task_metrics.model_config[model_idx - 1]
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

# TODO: Update/check for consistency with updated pre-scheduler
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
    global_seed, model_config = load_model_config(task_metrics, model_idx)
    train_config, dataset = reload_train_config(dir_tree, **training_config_overrides)

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
        logger.info("Architecture Training failed.")
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
        del model  # Release memory
        dir_tree.model_idx = None  # Ensure that previously written data cannot be overwritten

    ## Clean up memory before terminating task
    del task_metrics
    del dir_tree


if __name__ == "__main__":

    # Setup this module's logger
    # fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(fmt)
    # _log.addHandler(ch)
    # _log.setLevel(logging.INFO)
    #
    # sched_utils._log.addHandler(ch)
    # sched_utils._log.setLevel(logging.INFO)

    ## Parse CLI
    args = argument_parser().parse_args()

    # train_config_overrides = parse_training_overrides(args)
    train_config_overrides = get_tranining_config_from_args(args)

    workerid = args.workerid + args.workerid_offset
    worker_config = sched_utils.WorkerConfig(worker_id=workerid, portfolio_dir=args.portfolio_dir)
    worker_config.load_portfolio()

    logdir: Path = args.rootdir / "worker_logs"
    logdir.mkdir(exist_ok=True, parents=False)
    logger = naslib_logging.setup_logger(str(logdir / f"{workerid}.log"), name='tabular_sampling')
    if args.debug:
        # _log.setLevel(logging.DEBUG)
        # sched_utils._log.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    for taskid, model_idx, basedir in worker_config.iterate_portfolio(rootdir=args.rootdir):
        # conf = worker_config.portfolio.loc[(taskid, model_idx)]
        # basedir = args.rootdir / "-".join(["-".join([p, str(conf[p])]) for p in fidelity_params]) / "tasks"
        # run_task(basedir=basedir, taskid=taskid, )
        resume_work(basedir, taskid, model_idx, datadir=args.datadir, debug=args.debug, logger=logger,
                    **train_config_overrides)

