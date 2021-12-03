"""
A collection of utility functions to help verify the integrity of collected data and clean the checkpoints/metrics.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Generator, Tuple, Union, Iterable

import pandas as pd
import torch

from tabular_sampling.lib.utils import DirectoryTree, Checkpointer, MetricLogger

_log = logging.getLogger(__name__)


def _verify_chkpt(pth: Path, map_location: Optional[Any] = None):
    """ Attempt to load a single checkpoint file specified by the given path. Returns True if the file can be read
    successfully. 'map_location' has been provided for compatibility with GPUs. """

    try:
        _ = torch.load(pth, map_location=map_location)
    except RuntimeError as e:
        _log.debug(f"Found corrupt checkpoint: {pth}\nError type: {type(e)}\nError description: {str(e)}")
        return False
    except Exception as e:
        _log.info(f"Failed to read checkpoint {pth} due to unhandled error of type {type(e)}, error description: "
                  f"{str(e)}")
    finally:
        return True


def _verify_model_chkpts(dtree: DirectoryTree, /, cleanup: Optional[bool] = False, map_location: Optional[Any] = None):
    """ Given a DirectoryTree which has been initialized with a particular taskid and model_idx, verifies the
    integrity of all checkpoints under this model. 'map_location' has been provided for compatibility with GPUs. """

    assert dtree.taskid is not None, "The directory tree has not been initialzed with a taskid."
    assert dtree.model_idx is not None, "The directory tree has not been initialzed with a model index."

    chkpt_pths = Checkpointer._get_sorted_chkpt_paths(dtree.model_checkpoints_dir)
    for pth in chkpt_pths:
        check = _verify_chkpt(pth)
        if not check and cleanup:
            os.remove(pth)


def _verify_metrics_log(pth: Path):
    """ Attempt to load a single metrics DataFrame file specified by the given path. Returns True if the file can be
    read successfully. """

    try:
        _ = pd.read_pickle(pth)
    except pickle.PickleError as e:
        _log.debug(f"Found corrupt metrics log: {pth}\nError type: {type(e)}\nError description: {str(e)}")
        return False
    except Exception as e:
        _log.info(f"Failed to read metrics log {pth} due to unhandled error of type {type(e)}, error description: "
                  f"{str(e)}")
    finally:
        return True


def _verify_model_metrics(dtree: DirectoryTree, /, cleanup: Optional[bool] = False):
    """ Given a DirectoryTree which has been initialized with a particular taskid and model_idx, verifies the
    integrity of all metric DataFrames under this model. """

    assert dtree.taskid is not None, "The directory tree has not been initialzed with a taskid."
    assert dtree.model_idx is not None, "The directory tree has not been initialzed with a model index."

    metric_pths = MetricLogger._get_sorted_metric_paths(dtree.model_metrics_dir)
    for pth in metric_pths:
        check = _verify_metrics_log(pth)
        if not check and cleanup:
            os.remove(pth)


def iterate_model_tree(basedir: Path, taskid: Optional[int] = None, model_idx: Optional[int] = None,
                       enumerate: Optional[bool] = False) -> \
        Union[Generator[DirectoryTree], Generator[Tuple[int, int, DirectoryTree]]]:
    """
    Iterates over a directory-tree model-wise. If only 'basedir' is given, iterates over every model of every task
    found at this base directory. If a taskid is also specified, iterates over every model of that specific task. If a
    model index is specified in addition to a taskid, iterates over only that single model's directory tree, i.e.
    yields a single directory tree. It is possible to iterate over every model corresponding to a specific model idx in
    every task found at 'basedir' by setting taskid to None and specifying a value for model_idx. Setting 'enumerate'
    to True causes the generator to also yield the current taskid and model_idx.
    """

    # Opening in read-only mode prevents accidentally creating directories that did not already exist.
    dtree = DirectoryTree(basedir=basedir, read_only=True)

    if taskid is not None:
        dtree.taskid = taskid
        task_dirs = [dtree.task_dir]
    else:
        task_dirs = dtree.existing_tasks
        if task_dirs is None:
            _log.info(f"No tasks found at {basedir}.")
            yield StopIteration

    for t in task_dirs:
        dtree.taskid = int(t.stem)

        if model_idx is not None:
            dtree.model_idx = model_idx
            model_dirs = [dtree.model_dir]
        else:
            model_dirs = dtree.existing_models
            if model_dirs is None:
                _log.info(f"No models found in task {dtree.taskid} at {basedir}.")
                continue

        for m in model_dirs:
            dtree.model_idx = int(m.stem)
            new_tree = DirectoryTree(basedir=dtree.basedir, taskid=dtree.taskid, model_idx=dtree.model_idx,
                                     read_only=dtree.read_only)
            yield (t, m, new_tree) if enumerate else new_tree


def clean_corrupt_files(basedir: Path, taskid: Optional[int] = None, model_idx: Optional[int] = None,
                        /, cleanup: Optional[bool] = False, map_location: Optional[Any] = None):
    """
    Attempts to read each checkpoint and metric dataframe file and deletes any that cannot be read. If only 'basedir'
    is given, all tasks and their models found at this base directory will be cleaned. If a taskid is also specified,
    only the models of that specific task are cleaned. If a model index is specified in addition to a taskid, only that
    model's files are cleaned. The flag 'cleanup' has been intentionally introduced as an extra user-level check before
    deleting files - this flag must be set to True to delete corrupted files, otherwise, only debug messages for
    corrupted files are generated. This also enables a "debug mode" usage of this function call, with 'cleanup' set to
    False. 'map_location' has been provided for compatibility with GPUs.
    """

    for dtree in iterate_model_tree(basedir=basedir, taskid=taskid, model_idx=model_idx, enumerate=False):
        _verify_model_chkpts(dtree, cleanup=cleanup, map_location=map_location)
        _verify_model_metrics(dtree, cleanup=cleanup)


def _check_metric_data_integrity(workdir: Path, chpts: Iterable[dict], metric_logs: Iterable[pd.DataFrame]):
    """ Performs the actual data integrity check for a single model config on behalf of
    check_metric_data_integrity(). """

    pass


def check_metric_data_integrity(workdir: Path, basedir: Path, taskid: Optional[int] = None,
                                model_idx: Optional[int] = None, /, cleanup: Optional[bool] = False,
                                map_location: Optional[Any] = None):
    """
    This utility checks each logged metric DataFrame's registered number of epochs against those of the checkpoint
    closest to its own timestamp. In most cases, the checkpoint will have an identical timestamp and both the
    checkpoint and the metrics log will have an identical number of epochs. In the cases when an older checkpoint than
    the metrics log must be loaded, it is likely that metric logs recorded after this one contain "false" data -
    data for epochs for which the checkpoint has been lost. In such cases, the probable epoch IDs will be recorded.
    When the next metrics log and corresponding checkpoint are loaded, it is checked if the number of false epochs
    remains consistent or not. In case this consistency is maintained, it is highly likely that all data can be fixed
    by removing the false epochs, so the metric logs are modified and saved accordingly. If inconsistent false epochs
    are found, all checkpoints and metric logs after the last known timestamp at which the checkpoints and metric logs
    were correlated are marked as corrupt.

    The directory 'workdir' is expected to be a write-enabled directory where the function call will save generated
    outputs, such as summaries of suspected corrupt files.
    """

    def load_chkpts(pths):
        for p in pths:
            yield Checkpointer._load_checkpoint(p, safe_load=True, map_location=map_location)

    def load_metric_logs(pths):
        for p in pths:
            yield MetricLogger._load_metrics_log(p, safe_load=True)

    for t, m, dtree in iterate_model_tree(basedir=basedir, taskid=taskid, model_idx=model_idx, enumerate=True):
        chkpt_pths = Checkpointer._get_sorted_chkpt_paths(dtree.model_checkpoints_dir, ascending=True)
        metric_pths = MetricLogger._get_sorted_metric_paths(dtree.model_metrics_dir, ascending=True)

        chkpts = load_chkpts(chkpt_pths)
        metric_logs = load_metric_logs(metric_pths)



