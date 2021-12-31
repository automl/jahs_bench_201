""" This script is intended to be used for cleaning up a raw metrics DataFrame and prepare it for being fed into the
surrogate model training. """

import logging
import pandas as pd
from pathlib import Path
from tabular_sampling.lib.postprocessing import metric_df_ops
import argparse

_log = logging.getLogger(__name__)

def parse_cli():
    parser = argparse.ArgumentParser("Clean up raw metrics data and prepare it for training a surrogate model.")
    parser.add_argument("--data_pth", type=Path, help="The path to the raw metrics DataFrame file. Must be either a full "
                                                  "path or a path relative to the current working directory.")
    parser.add_argument("--outdir", type=Path,
                        help="Path to a directory where the cleaned data is stored as a pandas DataFrame named "
                             "'cleaned_metrics.pkl.gz'. The directory must already exist.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="The number of epochs that a full run is expected to have lasted for. All runs that have "
                             "data for fewer than this many epochs will be considered incomplete "
                             "(consult --keep_incomplete_runs). For configs with data from more epochs than this, "
                             "the extra epochs' data will be removed.")
    parser.add_argument("--keep_incomplete_runs", action="store_true",
                        help="When this flag is given, data from incomplete runs will be not filtered out completely, "
                             "otherwise, all data for runs without the threshold number of epochs will be removed.")
    parser.add_argument("--keep_nan_configs", action="store_true",
                        help="When this flag is given, if there are any NaN values in any of the metrics for any given "
                             "epoch of a config, all data from that config will be removed. Otherwise, only the data "
                             "for the epochs with NaN values will be removed.")

    args = parser.parse_args()
    return args

def clean(data_pth: Path, outdir: Path, epochs: int, keep_incomplete_runs: bool = False, keep_nan_configs: bool = False):
    data: pd.DataFrame = pd.read_pickle(data_pth)
    outfile = outdir / "cleaned_metrics.pkl.gz"

    # Throw away excess epochs' data
    data = data.loc[data.index.get_level_values("Epoch") <= epochs]

    if keep_incomplete_runs:
        nepochs: pd.DataFrame = metric_df_ops.get_nepochs(df=data)
        nepochs = nepochs.reindex(index=data.index, method="ffill")
        valid: pd.Series = nepochs.nepochs <= epochs
        data = data.loc[valid]

    nan_check = data.isna().any(axis=1)
    nan_ind: pd.MultiIndex = data.loc[nan_check].index

    if keep_nan_configs:
        # Generalize the indices to be dropped to the entire config instead of just particular epochs.
        nan_ind = nan_ind.droplevel("Epoch").drop_duplicates()

    data = data.drop(nan_ind)
    features = data["model_config"]
    features.loc[:, "Epoch"] = features.index.get_level_values("Epoch")
    outputs = data[pd.MultiIndex.from_product([["train", "valid", "test"], ["duration", "loss", "acc"]])]
    outputs.columns = metric_df_ops.collapse_index_names(outputs.columns, nlevels=2)
    diagnostics = data["diagnostic"][["FLOPS", "latency"]]
    outputs.loc[:, diagnostics.columns] = diagnostics
    outputs.loc[:, "size_MB"] = data["metadata"]["size_MB"]

    clean_data = pd.concat({"features": features, "labels": outputs}, axis=1)

    assert outdir.exists() and outdir.is_dir()
    clean_data.to_pickle(outfile)


if __name__ == "__main__":
    args = parse_cli()
    clean(**vars(args))