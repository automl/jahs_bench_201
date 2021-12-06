import argparse
from enum import Enum
import logging
from pathlib import Path
import pandas as pd
import sys
from typing import Optional, Union, Generator, Tuple

from tabular_sampling.lib.core import utils
from tabular_sampling.lib.core import constants
from tabular_sampling.lib.postprocessing import verification

_log = logging.getLogger(__name__)
fidelity_parameters = ["N", "W", "Resolution"]

def generate_worker_portfolio(workerid: int, nworkers: int, configs_pth: Path) -> pd.DataFrame:
    configs: pd.DataFrame = pd.read_pickle(configs_pth)
    _log.debug(f"Found {configs.index.size} configurations to construct portfolio from.")

    assert isinstance(configs, pd.DataFrame), f"The configs path should lead to a DataFrame, was {type(configs)}."
    assert isinstance(configs.index, pd.MultiIndex), \
        f"The configs DataFrame's index has not been configured properly. It should be a MultiIndex, was " \
        f"{type(configs.index)}"
    assert constants.MetricDFIndexLevels.taskid.value in configs.index.names, \
        f"Configs DataFrame index has not been configured properly, could not find " \
        f"{constants.MetricDFIndexLevels.taskid.value}"
    assert constants.MetricDFIndexLevels.modelid.value in configs.index.names, \
        f"Configs DataFrame index has not been configured properly, could not find " \
        f"{constants.MetricDFIndexLevels.modelid.value}"

    portfolio = configs.loc[configs.index[workerid::nworkers], :]
    _log.debug(f"Created portfolio of {portfolio.index.size} configs.")
    return portfolio


def _iterate_portfolio(portfolio: pd.DataFrame) -> Generator[Tuple[Tuple[int, int], pd.Series], None, None]:
    """ Generator function that, given a portfolio in the form of a pandas DataFrame, iterates over the model
    identifiers and the model configuration. The identifiers are returned as (taskid, model_idx) and the configuration
    as a Pandas Series containing the various parameter names in the index. """

    taskid_pos = portfolio.index.names.index(constants.MetricDFIndexLevels.taskid.value)
    modelidx_pos = portfolio.index.names.index(constants.MetricDFIndexLevels.modelid.value)

    for i in range(portfolio.index.size):
        current_config = portfolio.index[i]
        model_id = int(current_config[taskid_pos]), int(current_config[modelidx_pos])
        yield model_id, portfolio.loc[current_config, :]


def basedir_from_fidelity(config: pd.Series):
    basedir = Path("-".join(["-".join([f, str(config[f])]) for f in fidelity_parameters])) / "tasks"
    return basedir


def clean_data(portfolio: pd.DataFrame, rootdir: Path):

    _log.info(f"Performing clean up operation on data stored in root directory {rootdir}.")

    for (taskid, model_idx), config in _iterate_portfolio(portfolio):
        basedir = basedir_from_fidelity(config)
        _log.debug(f"Performing clean up for task {taskid}, model {model_idx} in directory tree at {basedir}.")
        verification.clean_corrupt_files(rootdir / basedir, taskid=taskid, model_idx=model_idx, cleanup=True)

    _log.info("Finished clean up operation.")


def prune_data(portfolio: pd.DataFrame, rootdir: Path, backupdir: Path):

    _log.info(f"Performing data pruning operation on data stored in root directory {rootdir}.")

    for (taskid, model_idx), config in _iterate_portfolio(portfolio):
        basedir = basedir_from_fidelity(config)
        _log.debug(f"Pruning out-of-sync data for task {taskid}, model {model_idx} in directory tree at "
                   f"{rootdir / basedir}.")
        verification.check_metric_data_integrity(backup_dir=backupdir / basedir, basedir=rootdir / basedir,
                                                 taskid=taskid, model_idx=model_idx, cleanup=True)


def verify_data_integrity(portfolio: pd.DataFrame, rootdir: Path, backupdir: Path, nepochs: int):
    pass


modes_of_operation = ["cleanup", "prune", "verify"]


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=modes_of_operation,
                        help="Mode of operation for this script. There are three main modes of operation: 'cleanup', "
                             "'prune' and 'verify'. In most use-cases, this script should be run in each of these "
                             "modes exactly in the order of their appearance above. 'cleanup' performs a very simple "
                             "and quick check to remove all checkpoints and metrics logs that are unreadable. 'prune' "
                             "is more involved that 'cleanup' and checks the checkpoints and metrics logs for "
                             "consistency. Any such pairs of files that are found to not be in sync with each other, "
                             "as well as subsequent pairs that would be affected, are removed. If either of these two "
                             "operations results in changes to the stored data, the respective model's directory is "
                             "marked as needing verification. In such a case, the 'verify' mode of this script comes "
                             "into play. It loads each checkpoint as well as the next metrics log in chronologically "
                             "ascending order, performs 'nepochs' evaluations from that checkpoint onwards, and "
                             "compares the generated data against the logged metrics data to verify whether or not "
                             "this checkpoint and metric log are valid. If an inconsistency is found, all data files "
                             "starting from the faulty checkpoint are removed.")
    parser.add_argument("--rootdir", type=Path, default=Path().cwd(),
                        help="Path to the root directory where all the tasks' output was stored. Task-specific "
                             "sub-directories will be created here if needed. Note that this is NOT the same as the "
                             "'basedir' of a DirectoryTree object as multiple DirectoryTree objects will be created "
                             "as and when needed.")
    parser.add_argument("--backupdir", type=Path, default="/tmp/nashpo_benchmarks/backupdir",
                        help="For the modes 'prune' and 'verify' of this script, data is never automatically deleted. "
                             "Removal only refers to moving the faulty data to an identical directory tree in this "
                             "backup directory. It must then be manually deleted when it has been verified that the "
                             "files are truly no longer necessary.")
    parser.add_argument("--workerid", type=int,
                        help="An offset from 0 for this worker within the current job's allocation of workers. This "
                             "does NOT correspond to a single 'taskid' but rather the portfolio of configurations that "
                             "this worker will evaluate.")
    parser.add_argument("--workerid_offset", type=int, default=0,
                        help="An additional fixed offset from 0 for this worker that, combined with the value of "
                             "'--worker', gives the overall worker ID in the context of a job that has been split into "
                             "multiple smaller parts.")
    parser.add_argument("--nworkers", type=int,
                        help="The total number of workers that are expected to concurrently perform data verification "
                             "in this root directory. This is used to coordinate the workers such that there is no "
                             "overlap in their portfolios.")
    parser.add_argument("--datadir", type=Path,
                        help="The directory where all datasets are expected to be stored.")
    parser.add_argument("--configs", type=Path,
                        help="Path to a pickled Pandas DataFrame (*.pkl.gz) that contains the full configuration of "
                             "each known model that should be verified. The index of each such configuration should be "
                             "the unique tuple (taskid, model_idx), where the corresponding values will be used to "
                             "construct corresponding directory trees.")
    parser.add_argument("--nepochs", type=int, default=None,
                        help="Only used in the 'verify' mode of operation. The number of epochs of data that should be "
                             "generated and verified using each loaded checkpoint. This implies that, for example, in "
                             "a model with 5 pairs of checkpoints and metric logs that have already been verified to "
                             "be in sync, the first 4 checkpoints will each be loaded up, the model trained for "
                             "'nepochs' epochs, and the corresponding entries in the last 4 metrics logs will be "
                             "cross-checked.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")

    return parser


if __name__ == "__main__":

    # Setup this module's logger
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)
    _log.setLevel(logging.INFO)

    verification._log.addHandler(ch)
    verification._log.setLevel(logging.INFO)

    ## Parse CLI
    args = argument_parser().parse_args()

    if args.debug:
        _log.setLevel(logging.DEBUG)
        verification._log.setLevel(logging.DEBUG)

    workerid = args.workerid_offset + args.workerid
    portfolio = generate_worker_portfolio(workerid=workerid, nworkers=args.nworkers, configs_pth=args.configs)

    _log.info(f"Beginning worker {workerid + 1}/{args.nworkers}.")

    if args.mode == modes_of_operation[0]:
        clean_data(portfolio, rootdir=args.rootdir)
    elif args.mode == modes_of_operation[1]:
        prune_data(portfolio, rootdir=args.rootdir, backupdir=args.backupdir)
    else:
        raise NotImplementedError(f"No existing implementation for mode {args.mode}.")

    _log.info(f"Worker {workerid + 1}/{args.nworkers} finished.")
