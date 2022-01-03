""" This script is intended to be used for cleaning up a raw metrics DataFrame and prepare it for being fed into the
surrogate model training. """

import logging
import math

import pandas as pd
from pathlib import Path
from tabular_sampling.lib.postprocessing import metric_df_ops
from tabular_sampling.lib.core import constants
import argparse

_log = logging.getLogger(__name__)

def parse_cli():
    parser = argparse.ArgumentParser(
        "Clean up raw metrics data and prepare it for training a surrogate model. This includes anonymizing the index "
        "(replace taskid/model_idx tuples with a unique integer value), filtering out for the desired number of epochs "
        "within each model config, filtering out incomplete models (if needed), filtering out NaN values (if needed), "
        "and sub-sampling the data."
    )
    parser.add_argument("--data_pth", type=Path, help="The path to the raw metrics DataFrame file. Must be either a full "
                                                  "path or a path relative to the current working directory.")
    parser.add_argument("--outfile", type=Path,
                        help="Name of the output file (including path) where the cleaned data is stored as a pandas "
                             "DataFrame. The parent directory must already exist. A file extension may be added.")
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
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of all available configs to be included in the cleaned data. Sub-sampling "
                             "occurs on a per-model-config basis, i.e. all epochs' data for each model config is "
                             "considered one unit for the sub-sampling. The script attempts to maintain the ratio of "
                             "model configs present in each fidelity group.")

    args = parser.parse_args()
    return args


def subsample_df(df: pd.DataFrame, subsample_frac: float = 1.0) -> pd.DataFrame:
    fids = metric_df_ops.get_configs(df=df)[constants.fidelity_params]
    fids["check"] = True  # Just a low-cost placeholder for groupby count()
    g = fids.groupby(constants.fidelity_params)
    counts = g.count()
    configs = g.head(math.floor(counts.min() * subsample))
    idx1 = df.index.to_frame()
    idx2 = configs.index.to_frame()
    sel = idx1.join(idx2, idx2.index.names, rsuffix="r_")  # Expand idx2 to also include the relevant epoch
    sel = sel.notna().all(axis=1)  # Convert to a mask
    df = df.loc[sel]
    return df


def clean(data_pth: Path, outdir: Path, epochs: int, keep_incomplete_runs: bool = False,
          keep_nan_configs: bool = False, subsample: float = 1.0):
    data: pd.DataFrame = pd.read_pickle(data_pth)
    outfile = outdir / "cleaned_metrics.pkl.gz"

    # Throw away excess epochs' data
    data = data.loc[data.index.get_level_values("Epoch") <= epochs]

    if keep_incomplete_runs:
        nepochs: pd.DataFrame = metric_df_ops.get_nepochs(df=data)
        nepochs = nepochs.reindex(index=data.index, method="ffill")
        valid: pd.Series = nepochs.nepochs <= epochs
        data = data.loc[valid]

    # Handle NaN values
    nan_check = data.isna().any(axis=1)
    nan_ind: pd.MultiIndex = data.loc[nan_check].index

    if keep_nan_configs:
        # Generalize the indices to be dropped to the entire config instead of just particular epochs.
        nan_ind = nan_ind.droplevel("Epoch").drop_duplicates()

    data = data.drop(nan_ind)

    # Subsample the dataset if needed
    if subsample < 1.0:
        data = subsample_df(df=data, subsample_frac=subsample)

    # Anonymize indices
    data = data.unstack("Epoch")
    idx = data.index.to_frame(index=False).reset_index(drop=True)  # Retain a copy of the original index mappings
    data.reset_index(drop=True, inplace=True)  # Assign unique ModelIndex values to each config
    data.index.set_names("ModelIndex", inplace=True)
    data = data.stack("Epoch")  # Now the index only has 2 levels - ModelIndex and Epoch
    data.reset_index("Epoch", col_level=1, col_fill="Index", inplace=True)  # Make the Epoch value a column
    data.reset_index(col_level=1, col_fill="Index", inplace=True)  # The dataset is now in long-format

    features = data["model_config"]
    features.loc[:, "Epoch"] = data["Index"]["Epoch"]
    outputs = data[pd.MultiIndex.from_product([["train", "valid", "test"], ["duration", "loss", "acc"]])]
    outputs.columns = metric_df_ops.collapse_index_names(outputs.columns, nlevels=2)
    diagnostics = data["diagnostic"][["FLOPS", "latency"]]
    outputs.loc[:, diagnostics.columns] = diagnostics
    outputs.loc[:, "size_MB"] = data["metadata"]["size_MB"]

    clean_data = pd.concat({"index": data["Index"], "features": features, "labels": outputs}, axis=1)

    assert outdir.exists() and outdir.is_dir()
    clean_data.to_pickle(outfile)


if __name__ == "__main__":
    args = parse_cli()
    clean(**vars(args))