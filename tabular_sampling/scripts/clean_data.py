"""
This script is intended to be used for cleaning up a raw metrics DataFrame and prepare it for being fed into the
surrogate model training. It, for example, filters out data from more than a specific number of epochs as well as can
be used to control whether or not data from configs with NaN values is to be included. This procedure must also account
for preventing a number of biases from being introduced into the dataset, such as sub-sampling the dataset from only a
sample of all available fidelity groups, thus destroying the underlying sampling distribution.

Workflow:
1.  Convert the data to long-format in order to make it easier to work with.
2.  Assign anonymized indices to each unique model config, as previously identified by the combination of "taskid" and
     "model_idx". Maintain a separate mapping of this.
3.  Assign a unique index to each fidelity group present in the model. Maintain a separate mapping of this.
4.  Rearrange the rows of the dataset such that the sampling distribution becomes agnostic to sub-indexing of the
     anonymized model index in terms of the fidelity group.
4a. Save the completed long-format data in order to avoid repeating the above mappings and ease maintenance of
     consistency across multiple data sub-sampling calls.
5.  Filter out model configs' data as per the limits on the number of epochs - throw away excess data, check if configs
     with too few epochs are to be retained.
6.  Filter out model configs' data as per the desirability of NaN values, divergence, etc.
7.  Perform config-wise sub-sampling, if needed.
8.  Save the resultant dataset in a long-format usable for surrogate training.

"""

import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from tabular_sampling.lib.core import constants
from tabular_sampling.lib.postprocessing import metric_df_ops

_log = logging.getLogger(__name__)


def parse_cli():
    parser = argparse.ArgumentParser(
        "Clean up raw metrics data and prepare it for training a surrogate model. This includes anonymizing the index "
        "(replace taskid/model_idx tuples with a unique integer value), filtering out for the desired number of epochs "
        "within each model config, filtering out incomplete models (if needed), filtering out NaN values (if needed), "
        "and sub-sampling the data."
    )
    parser.add_argument("--data_pth", type=Path,
                        help="The path to the raw metrics DataFrame file. Must be either a full "
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
                        help="When this flag is not given, if there are any NaN values in any of the metrics for any "
                             "given epoch of a config, all data from that config will be removed. Otherwise, only the "
                             "data for the epochs with NaN values will be removed. This is useful in certain cases, "
                             "e.g. when divergent configs' accuracies have been labelled with NaNs and all epochs "
                             "preceding the divergent epoch should be included.")
    parser.add_argument("--subsample", type=float, default=1.0,
                        help="Fraction of all available configs to be included in the cleaned data. Sub-sampling "
                             "occurs on a per-model-config basis, i.e. all epochs' data for each model config is "
                             "considered one unit for the sub-sampling. The script attempts to maintain the ratio of "
                             "model configs present in each fidelity group.")

    args = parser.parse_args()
    return args


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts an input DataFrame in the original metrics storage format to a long-format one. The returned DataFrame
    has three top-level columns: "sampling_index", "features", and "labels". SamplingIndex has the sub-columns "taskid"
    and "model_idx" and the corresponding values from the original index. The values of the "Epoch" level of the
    original index are moved underneath "features" instead in the sub-column "epoch" alongside all the sub-columns of
    "model_config" from the original DataFrame. All the metrics in the original DataFrame (with a name <metric>)
    recorded under a <set>: one of "train", "valid", and "test", are moved to the sub-column "<set>-<metric>". The
    sub-columns "FLOPS" and "latency" of "diagnostic" are also moved underneath "labels". Finally, "size_MB" from
    "metadata" is moved underneath "labels". The index itself is set to a simple RangeIndex. """

    # Separate out the features
    features = df["model_config"]
    features.loc[:, "epoch"] = df.index.get_level_values("Epoch")

    # Separate out the labels
    labels = df[["train", "valid", "test"]]
    labels.index = metric_df_ops.collapse_index_names(df.columns)
    diagnostics = df["diagnostic"][["FLOPS", "latency"]]
    labels.loc[:, diagnostics.columns] = diagnostics
    labels.loc[:, "size_MB"] = df["metadata"]["size_MB"]

    # Extract the original sampling index
    sampling_index = df.index.to_frame(index=False)
    features.index = sampling_index.index
    labels.index = sampling_index.index
    sampling_index.drop("Epoch", axis=1, inplace=True)

    combined = pd.concat({
        "sampling_index": sampling_index,
        "features": features,
        "labels": labels
    }, axis=1)

    return combined


def identify_model_configs(df: pd.DataFrame, inplace=False) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Given a DataFrame in long-format, as generated by ´to_long_format()´, return two Series. The first Series maps each
    unique value from the column sampling_index of the input DataFrame to a unique integer ID, called model ID. The
    second maps each index value of the input DataFrame to the corresponding model ID. This second series can thus
    be used to group the rows according to individual model configs. If ´inplace=True`, only the former is returned
    whereas the input DataFrame is modified in-place by adding a new column named "model_ID" under "sampling_index" and
    None is returned for the latter.

    :param df:
    :return:
    """

    sampling_index: pd.DataFrame = df.sampling_index
    sampling_index_to_model_id = sampling_index.drop_duplicates().reset_index(drop=True)
    sampling_index_to_model_id.index.set_names(["model_ID"], inplace=True)
    sampling_index_to_model_id.reset_index(drop=False, inplace=True)

    new_index = sampling_index.merge(sampling_index_to_model_id)

    if inplace:
        df.loc[:, ("sampling_index", "model_ID")] = new_index["model_ID"]
        return sampling_index_to_model_id, None
    else:
        return sampling_index_to_model_id, new_index


def identify_fidelity_groups(df: pd.DataFrame, fids: List[str], inplace=False) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Given a DataFrame in long-format, as generated by ´to_long_format()´, return two Series. The first Series maps each
    unique combination of values from the sub-columns under "features" identified by ´fids´ of the input DataFrame to a
    unique integer ID, called fidelity ID. The second maps each index value of the input DataFrame to the corresponding
    fidelity ID. This second series can thus be used to group the rows according to individual fidelity groups. If
    ´inplace=True`, only the former is returned whereas the input DataFrame is modified in-place by adding a new column
    named "fidelity_ID" under "sampling_index" and None is returned for the latter.

    :param df:
    :param fids: list of str
        A list of strings corresponding to the names of the fidelity parameters under "model_config" of the original
        metrics DataFrame and under "features" of the corresponding long-format DataFrame.
    :return:
    """

    fidelity_groups: pd.DataFrame = df.features[fids]
    fidelity_group_to_fidelity_id = fidelity_groups.drop_duplicates().reset_index(drop=True)
    fidelity_group_to_fidelity_id.index.set_names(["fidelity_ID"], inplace=True)
    fidelity_group_to_fidelity_id.reset_index(drop=False, inplace=True)

    new_index = fidelity_groups.merge(fidelity_group_to_fidelity_id)

    if inplace:
        df.loc[:, ("features", "fidelity_ID")] = new_index["fidelity_ID"]
        return fidelity_group_to_fidelity_id, None
    else:
        return fidelity_group_to_fidelity_id, new_index


def subsample_df(df: pd.DataFrame, subsample_frac: float = 1.0) -> pd.DataFrame:
    _log.info(f"Sub-sampling factor: {subsample_frac}")
    # Figure out which models, irrespective of how many epochs they have, should be selected
    fids = metric_df_ops.get_configs(df=df)[list(constants.fidelity_params)]
    fids.loc[:, "check"] = True  # Just a low-cost placeholder for groupby count()
    g = fids.groupby(list(constants.fidelity_params))
    counts = g.count()
    min_sample_size = math.floor(counts.min() * subsample_frac)
    _log.info(f"Reducing number of configs per fidelity group from an average of {counts.mean()} to {min_sample_size}.")
    configs = g.head(min_sample_size)

    # Create some placeholder dataframes for the two sets of indices we are concerned with - the full index of
    # available data, including epoch numbers, and the subset of models we want to select
    idx1 = pd.DataFrame(index=df.index)
    idx1.loc[:, "check"] = True
    idx2 = pd.DataFrame(index=configs.index)
    idx2.loc[:, "check"] = True

    # Effectively, expand idx2 to also include all the relevant epoch numbers for each model
    sel = idx1.join(idx2, idx2.index.names, rsuffix="_r")
    sel = sel.notna().all(axis=1)  # Convert to a mask

    # Select the relevant subset of rows
    df = df.loc[sel]
    _log.info(f"Sub-sampling successful.")
    return df


def interleave_fidelitywise(df: pd.DataFrame, fids: List[str]) -> pd.DataFrame:
    """
    In order to ensure that shuffling and splitting operations don't inadvertently introduce biases to the data, it is
    important to interleave the model configs of each fidelity group such that they are more or less uniformly
    distributed along the rows of the DataFrame. This operation should precede any sort of sub-sampling or re-sampling
    in order to maintain the legitimacy of any comparisons made on subsets of the whole dataset. Alternatively,
    comparisons on subsets of the whole dataset are only valid for subsets generated after this operation.
    :param df:
    :param fids:
        A list of strings denoting the column names corresponding to the fidelity parameters
    :return:
        A dataframe with the same data as df but re-arranged such that the distribution of model configs along the
        rows is agnostic to index-based sampling.
    """

    fidelities: pd.DataFrame = df["model_config"][fids]
    unique_fidelities = pd.MultiIndex.from_frame(fidelities).unique()
    unique_fidelities.to_frame(index=False)
    fidelity_index = pd.DataFrame(list(range(unique_fidelities.size)), index=unique_fidelities,
                                  columns=["FidelityIndex"])
    g = df["model_config"].groupby(fids)
    spare_cols = df["model_config"].columns.difference(fids)
    fidelity_counts = g[[spare_cols[0]]].count()


def clean(data_pth: Path, outfile: Path, epochs: int, keep_incomplete_runs: bool = False,
          keep_nan_configs: bool = False, subsample: float = 1.0):
    data: pd.DataFrame = pd.read_pickle(data_pth)
    outfile = outfile.parent / f"{outfile.name}.pkl.gz"

    _log.info("Throwing away excess epochs' data.")
    data = data.loc[data.index.get_level_values("Epoch") <= epochs]

    if keep_incomplete_runs:
        nepochs: pd.DataFrame = metric_df_ops.get_nepochs(df=data)
        nepochs = nepochs.reindex(index=data.index, method="ffill")
        valid: pd.Series = nepochs.nepochs <= epochs
        data = data.loc[valid]

    _log.info("Handling NaN values.")
    nan_check = data.isna().any(axis=1)
    nan_ind: pd.MultiIndex = data.loc[nan_check].index

    if keep_nan_configs:
        # Generalize the indices to be dropped to the entire config instead of just particular epochs.
        nan_ind = nan_ind.droplevel("Epoch").drop_duplicates()

    data = data.drop(nan_ind)

    _log.info("Subsampling the dataset if needed.")
    if subsample < 1.0:
        data = subsample_df(df=data, subsample_frac=subsample)

    _log.info("Anonymizing indices.")
    data = data.unstack("Epoch")
    _ = data.index.to_frame(index=False).reset_index(drop=True)  # Retain a copy of the original index mappings
    data.reset_index(drop=True, inplace=True)  # Assign unique ModelIndex values to each config
    data.index.set_names("ModelIndex", inplace=True)
    data = data.stack("Epoch")  # Now the index only has 2 levels - ModelIndex and Epoch
    data.reset_index("Epoch", col_level=1, col_fill="Groups", inplace=True)  # Make the Epoch value a column
    data.reset_index(col_level=1, col_fill="Groups", inplace=True)  # The dataset is now in long-format

    _log.info("Building dataset.")
    features = data["model_config"]
    features.loc[:, "Epoch"] = data["Groups"]["Epoch"]
    outputs = data[pd.MultiIndex.from_product([["train", "valid", "test"], ["duration", "loss", "acc"]])]
    outputs.columns = metric_df_ops.collapse_index_names(outputs.columns, nlevels=2)
    diagnostics = data["diagnostic"][["FLOPS", "latency"]]
    outputs.loc[:, diagnostics.columns] = diagnostics
    outputs.loc[:, "size_MB"] = data["metadata"]["size_MB"]

    clean_data = pd.concat({"groups": data["Groups"][["ModelIndex"]], "features": features, "labels": outputs}, axis=1)

    _log.info(f"Saving cleaned dataset, shape: {clean_data.shape}.")
    clean_data.to_pickle(outfile)

    _log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    cli_args = parse_cli()
    clean(**vars(cli_args))
