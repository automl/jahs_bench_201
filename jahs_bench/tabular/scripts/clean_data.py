"""
This script is intended to be used for cleaning up a raw metrics DataFrame and prepare it for being fed into the
surrogate model training. It, for example, filters out data from more than a specific number of epochs as well as can
be used to control whether or not data from configs with NaN values is to be included. This procedure must also account
for preventing a number of biases from being introduced into the dataset, such as sub-sampling the dataset from only a
sample of all available fidelity groups, thus destroying the underlying sampling distribution.

Workflow:
1.  Convert the data to wide-format in order to make it easier to work with.
2.  Assign anonymized indices to each unique model config, as previously identified by the combination of "taskid" and
     "model_idx". Maintain a separate mapping of this.
3.  Assign a unique index to each fidelity group present in the model. Maintain a separate mapping of this.
4.  Save the completed wide-format data in order to avoid repeating the above mappings and ease maintenance of
     consistency across multiple data sub-sampling calls.
5.  Filter out model configs' data as per the limits on the number of epochs - throw away excess data, check if configs
     with too few epochs are to be retained.
6.  Filter out model configs' data as per the desirability of NaN values, divergence, etc.
7.  Perform config-wise sub-sampling, if needed.
8.  Save the resultant dataset in a wide-format usable for surrogate training.

"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import sklearn.model_selection

from jahs_bench.tabular.lib.postprocessing import metric_df_ops

_log = logging.getLogger(__name__)


def parse_cli():
    parser = argparse.ArgumentParser(
        "Clean up raw metrics data and prepare it for training a surrogate model. This includes anonymizing the index "
        "(replace taskid/model_idx tuples with a unique integer value), filtering out for the desired number of epochs "
        "within each model config, filtering out incomplete models (if needed), filtering out NaN values (if needed), "
        "and sub-sampling the data."
    )
    parser.add_argument("--data_pth", type=Path,
                        help="The path to the raw metrics DataFrame file. Must be either a full path or a path "
                             "relative to the current working directory. The name (and file extension) of this file is "
                             "used to save the wide-format version at 'outdir'. When '--wide' is given, this file "
                             "should instead be the aforementioned wide-format version saved during a previous "
                             "iteration of this script. This must be a .pkl.gz file containing a pandas DataFrame.")
    parser.add_argument("--wide", action="store_true",
                        help="When this flag is given, it is assumed that the file specified by 'data_pth' was "
                             "generated during a previous iteration of this script and is already in wide-format. This "
                             "saves computation time as well as avoids consistency pitfalls when generating multiple "
                             "versions of the same dataset.")
    parser.add_argument("--outdir", type=Path, default=Path().cwd(),
                        help="Path to the directory where the cleaned data is to be stored as a pandas DataFrame. The "
                             "parent directory must already exist. Default: current working directory.")
    parser.add_argument("--output_filename", type=str, default="cleaned_metrics",
                        help="The name of the file in which the cleaned data will be stored. Appropriate file "
                             "extensions will automatically be added. Default: cleaned_metrics.")
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


def to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts an input DataFrame in the original metrics storage format to a wide-format one. The returned DataFrame
    has three top-level columns: "sampling_index", "features", and "labels". SamplingIndex has the sub-columns "taskid"
    and "model_idx" and the corresponding values from the original index. The values of the "Epoch" level of the
    original index are moved underneath "features" instead in the sub-column "epoch" alongside all the sub-columns of
    "model_config" from the original DataFrame. All the metrics in the original DataFrame (with a name <metric>)
    recorded under a <set>: one of "train", "valid", and "test", are moved to the sub-column "<set>-<metric>". The
    sub-columns "FLOPS" and "latency" of "diagnostic" are also moved underneath "labels". Finally, "size_MB" from
    "metadata" is moved underneath "labels". The index itself is set to a simple RangeIndex. """

    # Separate out the features
    features = df.loc[:, "model_config"]
    features.loc[:, "epoch"] = df.index.get_level_values("Epoch")

    # Separate out the labels
    labels = df.loc[:, ["train", "valid", "test"]]
    labels.columns = metric_df_ops.collapse_index_names(labels.columns, nlevels=2)
    diagnostics = df.loc[:, ("diagnostic", ["FLOPS", "latency", "runtime"])]
    labels.loc[:, diagnostics.columns.get_level_values(1)] = diagnostics.values
    labels.loc[:, "size_MB"] = df.loc[:, ("metadata", "size_MB")]

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
    Given a DataFrame in wide-format, as generated by ´to_wide_format()´, return two Series. The first Series maps each
    unique value from the column sampling_index of the input DataFrame to a unique integer ID, called model ID. The
    second maps each index value of the input DataFrame to the corresponding model ID. This second series can thus
    be used to group the rows according to individual model configs. If ´inplace=True`, only the former is returned
    whereas the input DataFrame is modified in-place by adding a new column named "model_ID" under "sampling_index" and
    None is returned for the latter.

    :param df: pandas DataFrame
    :param inplace: bool
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
    Given a DataFrame in wide-format, as generated by ´to_wide_format()´, return two Series. The first Series maps each
    unique combination of values from the sub-columns under "features" identified by ´fids´ of the input DataFrame to a
    unique integer ID, called fidelity ID. The second maps each index value of the input DataFrame to the corresponding
    fidelity ID. This second series can thus be used to group the rows according to individual fidelity groups. If
    ´inplace=True`, only the former is returned whereas the input DataFrame is modified in-place by adding a new column
    named "fidelity_ID" under "sampling_index" and None is returned for the latter.

    :param df: pandas DataFrame
    :param fids: list of str
        A list of strings corresponding to the names of the fidelity parameters under "model_config" of the original
        metrics DataFrame and under "features" of the corresponding wide-format DataFrame.
    :param inplace: bool
    :return:
    """

    fidelity_groups: pd.DataFrame = df.features[fids]
    fidelity_group_to_fidelity_id = fidelity_groups.drop_duplicates().reset_index(drop=True)
    fidelity_group_to_fidelity_id.index.set_names(["fidelity_ID"], inplace=True)
    fidelity_group_to_fidelity_id.reset_index(drop=False, inplace=True)

    new_index = fidelity_groups.merge(fidelity_group_to_fidelity_id)

    if inplace:
        df.loc[:, ("sampling_index", "fidelity_ID")] = new_index["fidelity_ID"]
        return fidelity_group_to_fidelity_id, None
    else:
        return fidelity_group_to_fidelity_id, new_index


def filter_nepochs(df: pd.DataFrame, nepochs: int, keep_incomplete: bool = False):
    """ Given a wide-format metrics DataFrame, filter out only up to the given number of epochs' worth of training data
    for each model config. If ´keep_incomplete=True´, model configs with fewer than ´nepochs´ worth of data are not
    thrown away. Note: It is assumed that the epochs in the wide-form table are 1-indexed, i.e. the first epoch's data
    has been marked by the value "1" for the field ´features.epoch´. """

    assert "sampling_index" in df.columns, "The input dataframe must be in a valid wide-format."
    assert "model_ID" in df.sampling_index.columns, "The input dataframe must be in a valid wide-format."

    _log.info("Throwing away excess epochs' data.")
    model_valid = df.features.epoch <= nepochs

    if keep_incomplete:
        return df.loc[model_valid]

    _log.info("Filtering out incomplete epochs' data.")
    existing_epoch_counts = df.groupby(("sampling_index", "model_ID"))[[("features", "epoch")]].max()
    existing_epoch_counts.columns = ["valid"]
    existing_epoch_counts.index.rename("model_ID", inplace=True)  # Get rid of the extra index level name

    models_complete = (existing_epoch_counts >= nepochs)
    models_complete = df.sampling_index.join(models_complete, on="model_ID")["valid"]
    df = df.loc[pd.concat([model_valid, models_complete], axis=1).all(axis=1)]

    return df


def subsample_df(df: pd.DataFrame, subsample_frac: float = 1.0) -> pd.DataFrame:
    """ Given a metrics DataFrame in wide-format, with uniquely identifiable model IDs and fidelity IDs, selects a
    specified fraction of all available model configs' data (regardless of number of epochs in each) while maintaining
    the sampling distribution across fidelity groups. """

    if subsample_frac > 1.:
        raise ValueError(f"Sub-sampling only supported for a value in the range (0., 1.], was given {subsample_frac}.")
    elif subsample_frac == 1.:
        return df

    _log.info(f"Sub-sampling factor: {subsample_frac}, original dataset size: {df.shape[0]}")

    # Figure out which models, irrespective of how many epochs they have, should be selected
    index = df.loc[:, ("sampling_index", slice(None))].droplevel(0, axis=1)
    model_ids = index.loc[:, "model_ID"]
    fidelity_groups = index.loc[:, "fidelity_ID"]

    nunique_models = model_ids.drop_duplicates().shape[0]
    nunique_fidelity_groups = fidelity_groups.drop_duplicates().shape[0]
    counts = index.groupby("fidelity_ID").count()
    _log.info(f"Original dataset has a total of {nunique_models} unique model configs spread over "
              f"{nunique_fidelity_groups} unique fidelity groups, averaging to {counts.mean().mean():.2f} data points "
              f"in each group.")

    n_splits = int(1 / subsample_frac)
    # For shuffle=False, StratifiedGroupKFold is deterministic w.r.t. an invariant dataset (i.e. the features, labels
    # and groups must remain the same)
    splitter = sklearn.model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    _, subsample_split = next(splitter.split(index, fidelity_groups, groups=model_ids))

    df = df.iloc[subsample_split]
    index = df.loc[:, ("sampling_index", slice(None))].droplevel(0, axis=1)
    model_ids = index.loc[:, "model_ID"]
    fidelity_groups = index.loc[:, "fidelity_ID"]

    nunique_models = model_ids.drop_duplicates().shape[0]
    nunique_fidelity_groups = fidelity_groups.drop_duplicates().shape[0]
    counts = index.groupby("fidelity_ID").count()
    _log.info(f"Sub-sampled dataset has a total of {df.shape[0]} data points, a total of {nunique_models} unique model "
              f"configs spread over {nunique_fidelity_groups} unique fidelity groups, averaging to "
              f"{counts.mean().mean():.2f} data points in each group.")
    return df


def clean(data_pth: Path, outdir: Path, output_filename: str, epochs: int, wide: bool = False,
          keep_incomplete_runs: bool = False, keep_nan_configs: bool = False, subsample: float = 1.0):
    assert outdir.exists() and outdir.is_dir()
    outfile = outdir / f"{output_filename}.pkl.gz"

    _log.info(f"Loading {'wide-format' if wide else 'raw'} metrics data from {data_pth}")
    data: pd.DataFrame = pd.read_pickle(data_pth)
    raw_shape = data.shape
    _log.info(f"Read metrics DataFrame of shape {raw_shape}.")

    if not wide:
        _log.info("Converting raw data to wide-format table.")
        data = to_wide_format(data)
        sampling_index_to_model_id, _ = identify_model_configs(data, inplace=True)
        fidelity_group_to_fidelity_id, _ = identify_fidelity_groups(data, fids=["N", "W", "Resolution"], inplace=True)
        data.sort_index(axis=1, inplace=True)
        wide_shape = data.shape
        assert wide_shape[0] == raw_shape[0], f"Unable to verify data integrity after conversion from raw table of " \
                                              f"shape {raw_shape} to wide table of shape {wide_shape}."
        _log.info("Successfully converted raw data to wide-format.")

        _log.info(f"Saving wide-format table to disk at {outdir}/wide_{data_pth.name}.")
        data.to_pickle(outdir / f"wide_{data_pth.name}", protocol=4)
        sampling_index_to_model_id.to_pickle(outdir / "sampling_index_to_model_id.pkl.gz", protocol=4)
        fidelity_group_to_fidelity_id.to_pickle(outdir / "fidelity_group_to_fidelity_id.pkl.gz", protocol=4)

    data = filter_nepochs(data, nepochs=epochs, keep_incomplete=keep_incomplete_runs)

    if not keep_nan_configs:
        # TODO: Handle datasets that don't include validation data
        _log.info("Handling NaN values.")
        nan_check = data.labels.isna().any(axis=1)
        nan_ind: pd.MultiIndex = data.loc[nan_check].index

        # if keep_nan_configs:
        #     # Generalize the indices to be dropped to the entire config instead of just particular epochs.
        #     nan_ind = nan_ind.droplevel("Epoch").drop_duplicates()

        data.drop(nan_ind, axis=0, inplace=True)

    data = subsample_df(df=data, subsample_frac=subsample)

    _log.info(f"Saving cleaned dataset, shape: {data.shape}.")
    data.to_pickle(outfile, protocol=4)

    _log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    cli_args = parse_cli()
    clean(**vars(cli_args))
