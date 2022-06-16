import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Generator, Tuple

import pandas as pd

from jahs_bench.tabular.lib.core import constants
from jahs_bench.tabular.lib.core import utils
from jahs_bench.tabular.lib.postprocessing import verification
from jahs_bench.tabular.clusterlib.prescheduler import fidelity_basedir_map

_log = logging.getLogger(__name__)
fidelity_parameters = ["N", "W", "Resolution"]
fidelity_to_str = {"N": lambda x: str(int(x)), "W": lambda x: str(int(x)), "Resolution": lambda x: str(x)}
worker_chkpt_subdir = "worker_chkpts"
cleanup_chkpt_name = "cleanup_progress.pkl.gz"
prune_chkpt_name = "prune_progress.pkl.gz"
verify_chkpt_name = "verify_progress.pkl.gz"

modes_of_operation = ["prepare", "cleanup", "prune", "verify"]


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


def subdir_from_fidelity(config: pd.Series):
    subdir = Path("-".join(["-".join([f, fidelity_to_str[f](config[f])]) for f in fidelity_parameters])) / "tasks"
    return subdir


def prepare_for_verification(basedir: Path, backupdir: Path):

    backupdir.mkdir(exist_ok=True, parents=True)
    try:
        backuptree = utils.DirectoryTree(basedir=backupdir)
    except FileExistsError:
        pass
    _log.debug(f"Preparing directory tree in {backuptree.basedir}.")
    for dtree in verification.iterate_model_tree(basedir=basedir, enumerate=False):
        try:
            backuptree.taskid = dtree.taskid
        except FileExistsError:
            pass
        try:
            backuptree.model_idx = dtree.model_idx
        except FileExistsError:
            pass

    _log.info("Finished creating backup directory structure.")



def clean_data(portfolio: pd.DataFrame, rootdir: Path, backupdir: Path, worker_chkpt_dir: Path, budget: int):
    _log.info(f"Performing clean up operation on data stored in root directory {rootdir}.")

    ## Prepare checkpoint
    chkpt_filename = worker_chkpt_dir / cleanup_chkpt_name
    done = None
    if chkpt_filename.exists():
        try:
            done = pd.read_pickle(chkpt_filename)
        except Exception as e:
            _log.info(f"Ran into an error while trying to read {chkpt_filename}: {str(e)}")
            done = None

    if done is None:
        done = pd.Series(False, index=portfolio.index).reorder_levels(
            [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value])
        done.to_pickle(chkpt_filename)

    ## Estimated time budget requirement
    max_time = 0.
    start = time.time()
    for (taskid, model_idx), config in _iterate_portfolio(portfolio):
        if budget < 1.2 * max_time:
            break

        if (taskid, model_idx) in done.index and done[(taskid, model_idx)]:
            continue

        subdir = subdir_from_fidelity(config)
        _log.debug(f"Performing clean up for task {taskid}, model {model_idx} in directory tree at {subdir}.")
        verification.clean_corrupt_files(basedir=rootdir / subdir, taskid=taskid, model_idx=model_idx, cleanup=True,
                                         backupdir=backupdir / subdir)

        ## Safely update checkpoint
        done[(taskid, model_idx)] = True
        shutil.copyfile(chkpt_filename, f"{chkpt_filename}.bak")
        done.to_pickle(chkpt_filename)

        ## Update duration estimate
        end = time.time()
        duration = end - start
        max_time = max(max_time, end - start)
        budget -= duration
        start = end

    _log.info(f"Finished clean up operation, max operation time: {max_time} seconds.")


def prune_data(portfolio: pd.DataFrame, rootdir: Path, backupdir: Path, worker_chkpt_dir: Path, budget: int):
    _log.info(f"Performing data pruning operation on data stored in root directory {rootdir}.")

    ## Prepare checkpoint
    chkpt_filename = worker_chkpt_dir / prune_chkpt_name
    done = None
    if chkpt_filename.exists():
        try:
            done = pd.read_pickle(chkpt_filename)
        except Exception as e:
            _log.info(f"Ran into an error while trying to read {chkpt_filename}: {str(e)}")
            done = None

    if done is None:
        done = pd.Series(False, index=portfolio.index).reorder_levels(
            [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value])
        done.to_pickle(chkpt_filename)

    ## Estimated time budget requirement
    max_time = 0.
    start = time.time()
    for (taskid, model_idx), config in _iterate_portfolio(portfolio):
        if budget < 1.2 * max_time:
            break

        if (taskid, model_idx) in done.index and done[(taskid, model_idx)]:
            continue

        subdir = subdir_from_fidelity(config)
        _log.debug(f"Pruning out-of-sync data for task {taskid}, model {model_idx} in directory tree at "
                   f"{rootdir / subdir}.")
        verification.check_metric_data_integrity(backup_dir=backupdir / subdir, basedir=rootdir / subdir,
                                                 taskid=taskid, model_idx=model_idx, cleanup=True)

        ## Safely update checkpoint
        done[(taskid, model_idx)] = True
        shutil.copyfile(chkpt_filename, f"{chkpt_filename}.bak")
        done.to_pickle(chkpt_filename)

        ## Update duration estimate
        end = time.time()
        duration = end - start
        max_time = max(max_time, end - start)
        budget -= duration
        start = end

    _log.info(f"Finished pruning operation, max operation time: {max_time} seconds.")


def verify_data_integrity(portfolio: pd.DataFrame, rootdir: Path, backupdir: Path, nepochs: int):
    pass


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=modes_of_operation,
                        help="Mode of operation for this script. There are four modes of operation: 'prepare', "
                             "'cleanup', 'prune' and 'verify'. In most use-cases, this script should be run in each of "
                             "these modes exactly in the order of their appearance above. 'prepare' runs a series of "
                             "preparatory procedures, such as creating the backup directory tree. 'cleanup' performs a "
                             "very simple and quick check to remove all checkpoints and metrics logs that are "
                             "unreadable. 'prune' is more involved than 'cleanup' and checks the checkpoints and "
                             "metrics logs for consistency. Any such pairs of files that are found to not be in sync "
                             "with each other, as well as subsequent pairs that would be affected, are removed. If "
                             "either of these two operations results in changes to the stored data, the respective "
                             "model's directory is marked as needing verification. In such a case, the 'verify' mode "
                             "of this script comes into play. It loads each checkpoint as well as the next metrics "
                             "log in chronologically ascending order, performs 'nepochs' evaluations from that "
                             "checkpoint onwards, and compares the generated data against the logged metrics data to "
                             "verify whether or not this checkpoint and metric log are valid. If an inconsistency is "
                             "found, all data files starting from the faulty checkpoint are removed.")
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
    parser.add_argument("--workerid-offset", type=int, default=0,
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
    parser.add_argument("--budget", type=int,
                        help="Time budget for each worker in seconds. This is used to approximate whether or not a "
                             "worker has enough time to perform one more operation without any risk to any data.")
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

    if not args.rootdir.exists():
        raise RuntimeError(f"The given root directory {args.rootdir} does not exist.")

    workerid = args.workerid_offset + args.workerid
    portfolio = generate_worker_portfolio(workerid=workerid, nworkers=args.nworkers, configs_pth=args.configs)
    worker_chkpt_dir: Path = args.rootdir / worker_chkpt_subdir / str(workerid)
    worker_chkpt_dir.mkdir(exist_ok=True, parents=True)

    _log.info(f"Worker {workerid + 1}: Beginning.")

    if args.mode == modes_of_operation[0]:
        basedirs = portfolio.loc[:, fidelity_parameters].apply(fidelity_basedir_map, axis=1).rename("basedir")
        basedirs = basedirs.unique()
        if workerid + 1 > basedirs.size:
            print(f"Worker {workerid + 1}: No preparatory work to do given that there are {basedirs.size} possible "
                  f"fidelity groupy defined in the given profile.")
            sys.exit(0)
        else:
            basedir = args.rootdir / basedirs[workerid]
            backupdir = args.backupdir / basedirs[workerid]
            _log.info(f"Worker {workerid + 1}: Preparing backup directory structure at {backupdir} for {basedir}.")

        prepare_for_verification(basedir=basedir, backupdir=backupdir)
    elif args.mode == modes_of_operation[1]:
        clean_data(portfolio, rootdir=args.rootdir, backupdir=args.backupdir, worker_chkpt_dir=worker_chkpt_dir,
                   budget=args.budget)
    elif args.mode == modes_of_operation[2]:
        prune_data(portfolio, rootdir=args.rootdir, backupdir=args.backupdir, worker_chkpt_dir=worker_chkpt_dir,
                   budget=args.budget)
    else:
        raise NotImplementedError(f"Worker {workerid + 1}: No existing implementation for mode {args.mode}.")

    _log.info(f"Worker {workerid + 1}: Finished.")
