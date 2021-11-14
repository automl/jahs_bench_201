"""
Collate all the data generated during training into one big pandas DataFrame. This script was designed specifically to
work with the directory structure defined by the class DirectoryTree. The individual DataFrame objects are expected to
contain all their data as columns with the appropriate metrics names. Theoretically, these can be anything, but the
code has been tested for the following for task and model metrics only:

tasks
-----
    - model_idx
    - model_config
    - global_seed
    - size_MB

models - [train/valid/test]
------
    - duration
    - forward_duration
    - data_load_duration
    - loss
    - acc
    - data_transfer_duration
    - backprop_duration

models - diagnostic
    - FLOPS
    - latency
    - runtime
    - cpu_percent
    - memory_ram
    - memory_swap
"""


from pathlib import Path
import pandas as pd
import json
import argparse
from tabular_sampling.lib import constants
from tabular_sampling.lib.constants import MetricDFIndexLevels
from tabular_sampling.lib.utils import DirectoryTree

parser = argparse.ArgumentParser("Collates all results from the DataFrames produced after successful NASLib training "
                                 "runs into a single large DataFrame.")
parser.add_argument("--basedir", type=Path, help="The base directory, same as what was used by the training script.")
parser.add_argument("--file", default=None, type=Path,
                    help="The desired output filename. The filename should not carry any extension as '.pkl.gz' will "
                         "be automatically appended to it, unless the extension is already '.pkl.gz'. "
                         "Default: <basedir>/data.pkl.gz")
args = parser.parse_args()

modelid_lvl = MetricDFIndexLevels.modelid.value

def get_latest_metrics(metric_dir: Path) -> pd.DataFrame:
    return max(
        metric_dir.rglob("*.pkl.gz"),
        key=lambda f: float(f.name.rstrip(".pkl.gz")),
        default=None
    )


task_metadata_columns = [m for m in constants.standard_task_metrics if m != "model_config"]
def task_df_hack(task_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Fix this hack at its source
    metadata = task_df[task_metadata_columns].droplevel(1, axis=1)
    model_configs = task_df["model_config"]
    return pd.concat({"model_config": model_configs, "metadata": metadata}, axis=1)


tree = DirectoryTree(basedir=args.basedir, read_only=True)
outfile: Path = tree.basedir / "data.pkl.gz" if args.file is None else args.file.resolve()
tasks = tree.existing_tasks
task_dfs = []

for i, t in enumerate(tasks, start=1):
    print(f"Processing task {i}/{len(tasks)}")
    tree.taskid = int(t.stem)
    latest_task_metrics = get_latest_metrics(tree.task_metrics_dir)
    task_metrics = pd.read_pickle(latest_task_metrics)
    task_metrics = task_df_hack(task_metrics)
    assert ("metadata", modelid_lvl) in task_metrics.columns, \
        f"Unable to process task metrics DataFrame that does not have a '{modelid_lvl}' column. Check DataFrame at " \
        f"{latest_task_metrics}."
    task_metrics = task_metrics.set_index(("metadata", modelid_lvl)).rename_axis([modelid_lvl], axis=0)
    models = tree.existing_models
    model_dfs = []
    for m in models:
        tree.model_idx = int(m.stem)
        latest_model_metrics = get_latest_metrics(tree.model_metrics_dir)
        if latest_model_metrics is None:
            continue
        model_metrics: pd.DataFrame = pd.read_pickle(latest_model_metrics)
        model_metrics.index = pd.MultiIndex.from_product([model_metrics.index, [tree.model_idx]],
                                                         names=model_metrics.index.names + [modelid_lvl])
        model_dfs.append(model_metrics)
    big_model_df = pd.concat(model_dfs, axis=0)
    task_df = big_model_df.join(task_metrics, on=modelid_lvl, how="left", lsuffix="_model", rsuffix="_task")
    task_df = task_df.assign(taskid=tree.taskid).set_index(constants.MetricDFIndexLevels.taskid.value, append=True)
    task_dfs.append(task_df)

big_task_df = pd.concat(task_dfs, axis=0)
big_task_df = big_task_df.reorder_levels(big_task_df.index.names[::-1], axis=0)
if len(outfile.name.split(".")) == 1:
    # Attach default file extension
    outfile = outfile.parent / f"{outfile.name}.pkl.gz"
big_task_df.to_pickle(outfile)

