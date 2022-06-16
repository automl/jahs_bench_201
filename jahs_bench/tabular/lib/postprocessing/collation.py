import json
import logging
import os
import traceback as tb
from pathlib import Path
from typing import Optional, Sequence, Iterable

import pandas as pd

from jahs_bench.tabular.lib.core import constants
from jahs_bench.tabular.lib.core.utils import DirectoryTree

_log = logging.getLogger(__name__)
metric_df_index_levels = [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value,
                          constants.MetricDFIndexLevels.epoch.value]


def sorted_metric_files(metric_dir: Path, descending: bool = True) -> Sequence[Path]:
    return sorted(
        metric_dir.rglob("*.pkl.gz"),
        key=lambda f: float(f.name.rstrip(".pkl.gz")),
        reverse=descending
    )


def safe_load_df(df_pkl_files: Iterable[Path], cleanup: bool = False) -> Optional[pd.DataFrame]:
    df = None
    for pkl in df_pkl_files:
        try:
            df: pd.DataFrame = pd.read_pickle(pkl)
        except Exception as e:
            _log.info(f"DataFrame {pkl} is likely corrupted, trying next file in sequence.")
            _log.debug(f"Could not load DataFrame from {pkl} because of the following error:\n{tb.format_exc()}")
            if cleanup:
                os.remove(pkl)
        else:
            _log.info(f"Loaded Pandas DataFrame from {pkl}.")
            break

    return df


task_metadata_columns = [m for m in constants.standard_task_metrics if m != "model_config"]
modelid_lvl = constants.MetricDFIndexLevels.modelid.value


def task_df_hack(task_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Fix this hack at its source
    metadata = task_df[task_metadata_columns].droplevel(1, axis=1)
    model_configs = task_df["model_config"]
    return pd.concat({"model_config": model_configs, "metadata": metadata}, axis=1)


# TODO: Implement 'save_summary'
# TODO: Fix implementation of 'anonymize' - we still want to retain the "Epoch" index level.
def collate_task_models(taskid: int, basedir: Optional[Path] = None, dtree: Optional[DirectoryTree] = None,
                        cleanup: bool = False, save_summary: bool = False, anonymize: bool = False) \
        -> Optional[pd.DataFrame]:
    """
    Iterates over all models evaluated by a particular task, collects the available model-level metrics, and
    returns a collated version of the metrics as a Pandas DataFrame. Either a directory 'basedir' that serves as the
    base directory for an appropriate directory tree or an instantiated DirectoryTree object 'dtree' should be given
    alongside the taskid.

    :param taskid: int
        The task id within all available tasks at the given directory tree to be handled.
    :param basedir: Path-like
        The base directory where the DirectoryTree object will be rooted. Either this or 'dtree' must be given.
    :param dtree: DirectoryTree
        An instantiated DirectoryTree object defining the directory tree where the required data will be sought.
    :param cleanup: bool
        When True, also removes stray files, such as error descriptions from repeated runs, that are no longer
        relevant. Default: False
    :param save_summary: bool
        When True, a summary of the current status of the models w.r.t. the expected training configuration is saved
        in the task directory. Default: False
    :param anonymize: bool
        When True, the original task id and model id of the metric data is removed from the final, returned DataFrame.
        Default: False
    :return: pd.DataFrame or None
        If no valid metric data could be loaded from the directory tree, most likely because none of the runs were
        completed enough to generate the metrics data or all generated data was corrupt, None is returned. Otherwise, a
        pandas DataFrame containing the collated metric data of all models evaluated under this task is returned.
    """

    assert basedir is not None or dtree is not None, "One of 'basedir' or 'dtree' must be given."

    if dtree is None:
        dtree = DirectoryTree(basedir=basedir, taskid=taskid, read_only=True)
    else:
        dtree.taskid = taskid

    if not dtree.task_dir.exists():
        raise RuntimeError(f"No existing data found for task id {dtree.taskid} in the directory tree rooted at "
                           f"{dtree.basedir}.")

    with open(dtree.task_config_file) as fp:
        task_config = json.load(fp)

    train_config = task_config["train_config"]
    expected_num_epochs = train_config["epochs"]

    latest_task_metrics = sorted_metric_files(dtree.task_metrics_dir, descending=True)
    task_metrics = safe_load_df(latest_task_metrics, cleanup=cleanup)

    if task_metrics is None:
        _log.debug(f"No valid task metrics DataFrame found for task id {dtree.taskid} in the directory tree "
                  f"rooted at {dtree.basedir}")
        return None

    task_metrics = task_df_hack(task_metrics)
    assert ("metadata", modelid_lvl) in task_metrics.columns, \
        f"Unable to process task metrics DataFrame that does not have the column ('metadata', '{modelid_lvl}')."

    task_metrics = task_metrics.set_index(("metadata", modelid_lvl)).rename_axis([modelid_lvl], axis=0)
    models = dtree.existing_models
    model_dfs = []
    for m in models:
        dtree.model_idx = int(m.stem)
        latest_model_metrics = sorted_metric_files(dtree.model_metrics_dir, descending=True)
        model_metrics = safe_load_df(latest_model_metrics, cleanup=cleanup)

        if model_metrics is None:
            _log.debug(f"No valid metric DataFrame found for model id {dtree.model_idx}.")
            continue

        model_metrics.index = pd.MultiIndex.from_product([model_metrics.index, [dtree.model_idx]],
                                                         names=model_metrics.index.names + [modelid_lvl])
        model_dfs.append(model_metrics)

    if not model_dfs:
        return None
    else:
        big_model_df = pd.concat(model_dfs, axis=0)
        task_df = big_model_df.join(task_metrics, on=modelid_lvl, how="left", lsuffix="_model", rsuffix="_task")
        if anonymize:
            task_df = task_df.reset_index(drop=True)
        else:
            # TODO: Make the 'taskid' name programmatically consistent in 'assign'.
            task_df = task_df.assign(taskid=dtree.taskid).set_index(constants.MetricDFIndexLevels.taskid.value,
                                                                    append=True)
            task_df = task_df.reorder_levels(metric_df_index_levels, axis=0)

        return task_df


# TODO: Implement 'save_summary'
def collate_tasks(basedir: Optional[Path] = None, dtree: Optional[DirectoryTree] = None, cleanup: bool = False,
                  save_summary: bool = False, anonymize: bool = True) -> Optional[pd.DataFrame]:
    """
    Collate the metrics of all evaluated models in all tasks under the given directory tree, as defined by either
    the specified base directory 'basedir' or an instance of DirectoryTree passed as 'dtree'. The flag 'cleanup', when
    True, also removes stray files, such as error descriptions from repeated runs, that are no longer relevant. When
    the flag 'save_summary' is True, a summary of the current status of all the tasks w.r.t. their expected training
    configurations is saved in the base directory.

    :param basedir: Path-like
        The base directory where the DirectoryTree object will be rooted. Either this or 'dtree' must be given.
    :param dtree: DirectoryTree
        An instantiated DirectoryTree object defining the directory tree where the required data will be sought.
    :param cleanup: bool
        When True, also removes stray files, such as error descriptions from repeated runs, that are no longer
        relevant. Default: False
    :param save_summary: bool
        When True, a summary of the current status of the models w.r.t. the expected training configuration is saved
        in the task directory. A job-level summary of all the tasks is also saved in the base directory. Default: False
    :param anonymize: bool
        When True, the original task id and model id of the metric data is removed from the final, returned DataFrame.
        Default: False
    :return: pd.DataFrame or None
        If no valid metric data could be loaded from the directory tree, most likely because none of the runs were
        completed enough to generate the metrics data or all generated data was corrupt, None is returned. Otherwise, a
        pandas DataFrame containing the collated metric data of all models and all tasks evaluated under this task is
        returned.
    """

    assert basedir is not None or dtree is not None, "One of 'basedir' or 'dtree' must be given."

    if dtree is None:
        dtree = DirectoryTree(basedir=basedir, read_only=True)

    tasks = dtree.existing_tasks
    if tasks is None:
        _log.info(f"Found no existing task data sub-directories in the directory tree rooted at {dtree.basedir}")
        return None

    tasks = list(map(lambda t: int(t.stem), tasks))
    ntasks = len(tasks)
    task_dfs = []

    for i, t in enumerate(tasks, start=1):
        _log.info(f"Processing task {i}/{ntasks}")
        task_df = collate_task_models(taskid=t, dtree=dtree, cleanup=cleanup, save_summary=save_summary,
                                      anonymize=anonymize)
        if task_df is None:
            _log.debug(f"No valid metric data found for task id {t} in the directory tree rooted at {dtree.basedir}")
            continue

        task_dfs.append(task_df)

    if not task_dfs:
        _log.info(f"Found no valid metric data for any of the tasks in the directory tree rooted at {dtree.basedir}")
        return None
    else:
        big_task_df = pd.concat(task_dfs, axis=0)
        big_task_df = big_task_df.reorder_levels(metric_df_index_levels, axis=0)

        return big_task_df
