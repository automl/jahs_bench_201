"""
A collection of utility functions to help verify the integrity of collected data and clean the checkpoints/metrics.
"""
import json
from functools import partial
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Generator, Tuple, Union, Iterable, List, Dict

import pandas as pd
import torch

from tabular_sampling.lib.core.utils import DirectoryTree, Checkpointer, MetricLogger

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


class MetricDataIntegrityChecker:
    dtree: DirectoryTree

    found_faults = False  # If faults have been found, there is a need to run a more intense data verification procedure
    consistent = True  # Should the data be inconsistent, there is no choice but to abandon it
    faulty_epochs = []  # The epoch nums found to be the cause of data faults
    verified_chkpts = []  # The checkpoint timestamps that have been verified to be consistent and fault-free
    verified_metric_logs = []  # The metrics log timestamps that have been verified to be consistent and fault-free
    fixed_metric_logs = []  # The metrics log timestamps that needed to be fixed
    invalid_chkpts = []  # The checkpoint timestamps that have been deemed inconsistent and should be deleted
    invalid_metric_logs = []  # The metrics log timestamps that have been deemed inconsistent and should be deleted

    __CHECKPOINT = Dict

    def __init__(self, dtree: DirectoryTree, enable_cleanup: bool = False):
        assert dtree.taskid is not None and dtree.model_idx is not None, \
            "The directory tree must be initialized fully with a task id and model index before being passed. The " \
            f"given tree had task id {dtree.taskid} and model index {dtree.model_idx}."
        self.dtree = dtree
        self.enable_cleanup = enable_cleanup

    # Convenience wrappers
    _load_chkpt = partial(Checkpointer._load_checkpoint, safe_load=True, map_location=map_location)
    _load_metric_log = partial(MetricLogger._load_metrics_log, safe_load=True)

    @property
    def chkpt_pths(self) -> Dict[float, __CHECKPOINT]:
        """ Dictionary mapping float timestamp to checkpoint file Path, sorted in descending order, i.e. most recent
        checkpoint first. """
        return {Checkpointer._extract_runtime_from_filename(p): p for p in
                Checkpointer._get_sorted_chkpt_paths(pth=self.dtree.model_checkpoints_dir, ascending=False)}

    @property
    def chkpt_timestamps(self) -> List[float]:
        """ The list of timestamps for this model's available checkpoints, in chronologically descending order. """
        return list(map(Checkpointer._extract_runtime_from_filename,
                        Checkpointer._get_sorted_chkpt_paths(pth=self.dtree.model_checkpoints_dir, ascending=False)))

    @property
    def metrics_log_pths(self) -> Dict[float, pd.DataFrame]:
        """ Dictionary mapping float timestamp to metrics log file Path, sorted in descending order, i.e. most recent
        log first. """
        return {MetricLogger._extract_runtime_from_filename(p): p for p in
                MetricLogger._get_sorted_metric_paths(pth=self.dtree.model_metrics_dir, ascending=False)}

    @property
    def metrics_log_timestamps(self) -> List[float]:
        """ The list of timestamps for this model's available metrics logs, in chronologically descending order. """
        return list(map(Checkpointer._extract_runtime_from_filename,
                        Checkpointer._get_sorted_chkpt_paths(pth=self.dtree.model_checkpoints_dir, ascending=False)))

    def save_results(self):
        """ Saves a JSON file in the model directory detailing the results of the verification procedure. """
        save_dict = {
            "found_faults": self.found_faults,
            "consistent": self.consistent,
            "faulty_epochs": self.faulty_epochs,
            "verified_chkpts": self.verified_chkpts,
            "verified_metric_logs": self.verified_metric_logs,
            "fixed_metric_logs": self.fixed_metric_logs,
            "invalid_chkpts": self.invalid_chkpts,
            "invalid_metric_logs": self.invalid_metric_logs,
        }
        with open(self.dtree.model_metrics_dir / "metric_logs_verification_results.json") as fp:
            json.dump(save_dict, fp)

    def _declare_invalid(self):
        """ Declares all of the current model's data to be invalid and saves the results to disk. """

        self.consistent = False
        self.invalid_chkpts = list(sorted(self.chkpt_pths.keys(), reverse=True))
        self.invalid_metric_logs = list(sorted(self.metrics_log_pths.keys(), reverse=True))
        save_results()

    def _iterate_pairs_over_timestamps(self, start: Optional[int] = 0, stop: Optional[int] = None,
                                       step: Optional[int] = 1, timestamps: bool = False, nepochs: bool = False) \
            -> Generator[Union[Tuple[__CHECKPOINT, pd.DataFrame], Tuple[__CHECKPOINT, pd.DataFrame, int, int],
            Tuple[__CHECKPOINT, pd.DataFrame, float], Tuple[__CHECKPOINT, pd.DataFrame, float, int, int]]]:
        """ Iterates over all the checkpoint timestamps in chronologically descending order and yields the
        corresponding checkpoint and the metrics log. If 'timestamps' is True, also returns the corresponding timestamp
        as a float. If 'nepochs' is True, also returns the #epochs in checkpoint and #epochs in metrics log. """

        timestamps = self.chkpt_timestamps
        chkpt_pths = self.chkpt_pths
        metrics_log_pths = self.metrics_log_pths

        for timestamp in timestamps[start:stop:step]:
            chkpt = load_chkpt(chkpt_pths[timestamp])
            metrics_log = load_metric_log(metrics_log_pths[timestamp])
            nepochs_chkpt = chkpt["epoch"]
            nepochs_metrics = metrics_log.index.size

            if timestamps and nepochs:
                yield chkpt, metrics_log, timestamp, nepochs_chkpt, nepochs_metrics
            elif timestamps:
                yield chkpt, metrics_log, timestamp
            elif nepochs:
                yield chkpt, metrics_log, nepochs_chkpt, nepochs_metrics
            else:
                yield chkpt, metrics_log

            del chkpt
            del metrics_log

    def _is_all_metric_data_clean(self) -> bool:

        gen = self._iterate_pairs_over_timestamps(timestamps=False, nepochs=True)
        for chkpt, metrics_log, nepochs_chkpt, nepochs_metrics in gen:

            if nepochs_chkpt != nepochs_metrics:
                _log.debug(f"Checkpoint {timestamp} of model at {self.dtree.model_dir} is not consistent in terms of "
                           f"number of epochs with its corresponding metric logs.")
                del gen
                return False
        return True

    @classmethod
    def _fix_metric_logs_faulty_epochs(cls, metrics_log: pd.DataFrame, faulty_epochs: Iterable[int]) -> pd.DataFrame:
        """ Removes the rows corresponding to indices listed in 'faulty_epochs' from 'metrics_log', re-indexes the
        DataFrame and returns it. """

        metrics_log: pd.DataFrame = metrics_log.drop(index=faulty_epochs)
        nepochs = metrics_log.index.size
        metrics_log.index = pd.Int64Index(list(range(nepochs))) + 1
        return metrics_log

    def check_model_metric_data_integrity(self):
        """
        Performs the actual data integrity check for a single model config on behalf of
        check_metric_data_integrity().
        How it works:
        1 - For each checkpoint, load the corresponding metric log, starting from the most recent.
        2 - If no inconsistency is found, move to (7).
        3 - If an inconsistency in the number of epochs is found, keep loading older checkpoints until they become
            consistent again. Mark this timestamp as t_const and the oldest inconsistent one as t_inconst.
        4 - Assume that the most recent metric log older than t_inconst is the one that caused the inconsistency.
            Therefore, deleting the corresponding epochs should remove all inconsistency.
        5 - Act on this assumption to try and fix all known inconsistent metric logs.
        6 - Start from 1 again.
        7 - If any inconsistencies were found, mark this model for cross-verification and delete all metric logs
            without corresponding checkpoint files.
        """

        ## Part 1, look for faulty data
        timestamps = self.chkpt_timestamps
        first_inconsistent_timestamp = max(timestamps) + 1.
        last_consistent_timestamp = -1
        n_faulty_epochs = 0

        # Iterate over all checkpoint/metrics log pairs by checkpoint timestamp and locate the more recent pair which
        # is consistent and the oldest pair which is not consistent. Consistency is defined as having a matching number
        # of registered epochs.
        gen = self._iterate_pairs_over_timestamps(timestamps=True, nepochs=True)
        for chkpt, metrics_log, timestamp, nepochs_chkpt, nepochs_metrics in gen:
            if nepochs_chkpt == nepochs_metrics:
                last_consistent_timestamp = max(timestamp, last_consistent_timestamp)
            elif nepochs_chkpt > nepochs_metrics:
                raise RuntimeError(f"Unexpected inconsistency type encountered in {self.dtree.model_dir} at timestamp "
                                   f"{timestamp}.")
            else:
                first_inconsistent_timestamp = min(timestamp, first_inconsistent_timestamp)
                n_faulty_epochs = nepochs_metrics - nepochs_chkpt

        if first_inconsistent_timestamp == -1.:
            # No inconsistent pairs were found. This model's data is currently clean.
            _log.debug(f"Data at {self.dtree.model_dir} verified to be consistent. Moving on to cleanup.")
            self._cleanup()
            return

        elif last_consistent_timestamp == -1.:
            # There is no clean data since we checked all available checkpoints and found no consistent pairs.
            _log.debug(f"Found irrecoverably inconsistent data at {self.dtree.model_dir}.")
            self._declare_invalid()
            return

        elif last_consistent_timestamp < first_inconsistent_timestamp:
            # This is probably a data fault that can be recovered from. Most likely, a checkpoint file was corrupted
            # and could not be read, so the training procedure loaded an older checkpoint but a newer metrics log and
            # appended the new metrics to the latter.
            self.found_faults = True
            self._locate_faults(first_inconsistent_timestamp, last_consistent_timestamp, n_faulty_epochs)

            # Do another pass of this procedure. This verifies if there are any more data faults and eventually cleans
            # up any stray files before exiting.
            self.check_model_metric_data_integrity()
        else:
            # This is some completely unexpected type of data corruption and we have no idea how to solve it.
            _log.debug(f"Found irrecoverably inconsistent data at {self.dtree.model_dir}.")
            self._declare_invalid()
            return


    def _locate_faults(self, first_inconsistent_timestamp: float, last_consistent_timestamp: float,
                       n_faulty_epochs: int):
        """ Attempts to locate a stray metrics log file that may have introduced inconsistencies in the logged data,
        and retroactively removes the faulty data. """

        chkpt_pths = self.chkpt_pths
        metrics_log_pths = self.metrics_log_pths

        timestamps = self.metrics_log_timestamps
        idx_first_inconst = timestamps.index(first_inconsistent_timestamp)
        idx_last_const = timestamps.index(last_consistent_timestamp)

        assert idx_last_const > idx_first_inconst, \
            "The first inconsistent pair cannot occur before the last consistent pair."

        source_inconsistency = timestamps[idx_first_inconst:idx_last_const][0]
        assert source_inconsistency <= first_inconsistent_timestamp, \
            "Expected the timestamps to be arranged in descending order."

        problematic_metrics_log = self._load_metric_log(metrics_log_pths[source_inconsistency])
        nepochs_source_inconst = problematic_metrics_log.index.size
        del problematic_metrics_log

        last_consistent_chkpt = self._load_chkpt(chkpt_pths[last_consistent_timestamp])
        nepochs_last_consistent = last_consistent_chkpt["epoch"]
        del last_consistent_chkpt

        if n_faulty_epochs != nepochs_source_inconst - nepochs_last_consistent:
            _log.info(f"Found irrecoverably inconsistent data at {self.dtree.model_dir}.")
            self._declare_invalid()
        else:
            # Try to fix inconsistencies in all checkpoints moving from this point onwards
            timestamps = self.chkpt_timestamps
            faulty_epochs = list(range(nepochs_last_consistent + 1, nepochs_last_consistent + n_faulty_epochs + 1))

            for timestamp in timestamps[idx_first_inconst:]:
                metrics_log = self._load_metric_log(metrics_log_pths[timestamp])
                metrics_log = self._fix_metric_logs_faulty_epochs(metrics_log, faulty_epochs=faulty_epochs)
                metrics_log.to_pickle(metrics_log_pths[timestamp])

        return

    def _cleanup(self):
        """ If the cleanup flag is set, removes all metrics logs for which a corresponding checkpoint does not exist.
        """

        if not self.enable_cleanup:
            return

        all_metrics = self.metrics_log_pths
        all_chkpts = self.chkpt_pths

        for k, v in all_metrics.items():
            if k not in all_chkpts:
                _log.debug(f"Removing stray metrics log file {v}")
                os.remove(v)


def check_metric_data_integrity(basedir: Path, taskid: Optional[int] = None, model_idx: Optional[int] = None, /,
                                cleanup: Optional[bool] = False, map_location: Optional[Any] = None):
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
    """

    for t, m, dtree in iterate_model_tree(basedir=basedir, taskid=taskid, model_idx=model_idx, enumerate=True):
        checker = MetricDataIntegrityChecker(dtree, enable_cleanup=False)
        checker.check_model_metric_data_integrity()

