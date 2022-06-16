"""
This is mostly a set of convenience wrappers around small sequences of operations on collated metric DataFrames that
make it easier to post-process the data in a semantic manner. Another advantage of using the convenience functions here
is that they handle a lot of boiler-plate code for implementing various checks and balances and ensuring consistent
outputs.

Expected structure of a single job's metric DataFrame:

index - pandas.MultiIndex, levels: [taskid, model_idx, Epoch]
columns - pandas.MultiIndex, level names: [MetricType, MetricName]
    where MetricType is generally [train, valid, test, diagnostic] and MetricName corresponds to values from
    tabular_sampling.lib.constants

"""

import argparse
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Iterable, Union, Optional, Any

import pandas as pd

from jahs_bench.tabular.lib.core.constants import MetricDFIndexLevels
from jahs_bench.tabular.search_space.constants import OP_NAMES, EDGE_LIST

_log = logging.getLogger(__name__)

model_ids_by = [MetricDFIndexLevels.taskid.value, MetricDFIndexLevels.modelid.value]


# fidelity_params=("N", "W")

# Common settings
# plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16})


def map_label_to_color(label: str, map: Dict, colors: Iterable):
    if not label in map:
        map[label] = next(colors)
    return map[label]


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, help="Path to the base directory where the relevant dataframe(s) "
                                                     "is/are stored as a file named data.pkl.gz.")
    args = parser.parse_args()
    return args


def load_metric_df(basedir: Path = None, df: pd.DataFrame = None) -> pd.DataFrame:
    assert basedir is not None or df is not None, "Either the 'basedir' should be a Path to load the data from " \
                                                  "or 'df' should be a pre-loaded metric data DataFrame."
    if df is None:
        # TODO: Make the metric df pickle file name a reusable constant
        df_fn = basedir / "metrics.pkl.gz"
        df: pd.DataFrame = pd.read_pickle(df_fn)
    else:
        assert isinstance(df, pd.DataFrame), f"This script was designed to only work with pandas DataFrames, not " \
                                             f"{type(df)}"

    return df


def _df_loader_wrapper(func):
    @wraps(func)
    def wrapper(basedir: Path = None, df: pd.DataFrame = None, *args, **kwargs) -> pd.DataFrame:
        df = load_metric_df(basedir=basedir, df=df)
        return func(basedir, df, *args, **kwargs)

    return wrapper


@_df_loader_wrapper
def get_runtimes(basedir: Path, df: pd.DataFrame, display: bool = False, reduce_epochs: bool = True,
                 extra_durations: list = None) -> pd.DataFrame:
    """
    Extract the runtimes of the various models.

    :param basedir:
    :param df:
    :param display: bool
        When True, logs the runtime statistics.
    :param reduce_epochs: bool
        When True, abstracts away the per-epoch metrics and generates runtimes on a per-model basis instead.
    :return:
    """

    datasets = ["train", "valid", "test"]
    duration_metrics = ["duration"] + (extra_durations if extra_durations is not None else [])
    runtime_metrics = [(ds, met) for ds in datasets for met in duration_metrics] + [("diagnostic", "runtime")]

    if reduce_epochs:
        d1 = df.loc[:, runtime_metrics[:-1]].groupby(model_ids_by).agg("sum")
        d2 = df.loc[:, runtime_metrics[-1:]].groupby(model_ids_by).agg("max")
        d1.columns = d1.columns.map("{0[0]}-{0[1]}".format)
        d2.columns = d2.columns.map("{0[1]}".format)
        runtimes_df = d1.join([d2])
    else:
        runtimes_df = df[runtime_metrics]

    if display:
        _log.info(f"Runtime statistics:\n{runtimes_df.describe()}")
    return runtimes_df


@_df_loader_wrapper
def get_nepochs(basedir: Path, df: pd.DataFrame, filter_epochs: int = -1) -> pd.DataFrame:
    assert isinstance(df.index, pd.MultiIndex), f"The input DataFrame index must be a MultiIndex, was {type(df.index)}"
    assert MetricDFIndexLevels.epoch.value in df.index.names, \
        f"Missing the level '{MetricDFIndexLevels.epoch.value}' for epochs in input DataFrame with MultiIndex index " \
        f"levels {df.index.names}."

    nepochs = df[df.columns.values[0]]
    nepochs = nepochs.groupby(model_ids_by).agg("count").to_frame(("nepochs"))

    if filter_epochs > 0:
        nepochs = nepochs[nepochs["nepochs"].where(nepochs["nepochs"] == filter_epochs).notna()]

    return nepochs


@_df_loader_wrapper
def get_configs(basedir: Path, df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df.index, pd.MultiIndex), f"The input DataFrame index must be a MultiIndex, was {type(df.index)}"
    assert MetricDFIndexLevels.epoch.value in df.index.names, \
        f"Missing the level '{MetricDFIndexLevels.epoch.value}' for epochs in input DataFrame with MultiIndex index " \
        f"levels {df.index.names}."
    assert all([l in df.index.names for l in model_ids_by]), \
        f"The input DataFrame index must include the levels {model_ids_by}, but had the levels {df.index.names}."

    confs = df["model_config"].xs(1, level=MetricDFIndexLevels.epoch.value)
    confs = confs.reorder_levels(model_ids_by, axis=0)

    return confs


@_df_loader_wrapper
def get_accuracies(basedir: Path, df: pd.DataFrame, include_validation: bool = False) -> pd.DataFrame:
    """
    Extract and return a DataFrame containing only the train, test and (if enabled) validation accuracy scores.
    """

    # outdir = basedir / "analysis"
    # outdir.mkdir(exist_ok=True, parents=True)

    assert isinstance(df.index, pd.MultiIndex), f"The input DataFrame index must be a MultiIndex, was {type(df.index)}"
    assert all([l in df.index.names for l in model_ids_by]), \
        f"The input DataFrame index must include the levels {model_ids_by}, but had the levels {df.index.names}."

    train_acc: pd.DataFrame = df[("train", "acc")].groupby(model_ids_by).agg("max").to_frame("train-acc")
    test_acc: pd.DataFrame = df[("test", "acc")].groupby(model_ids_by).agg("max").to_frame("test-acc")
    if include_validation:
        valid_acc: pd.DataFrame = df[("valid", "acc")].groupby(model_ids_by).agg("max").to_frame("valid-acc")
        acc_df = test_acc.join([train_acc, valid_acc])
    else:
        acc_df = test_acc.join([train_acc])

    return acc_df


@_df_loader_wrapper
def get_losses(basedir: Path, df: pd.DataFrame, include_validation: bool = False) -> pd.DataFrame:
    """
    Extract and return a DataFrame containing only the train, test and (if enabled) validation losses.
    """

    # outdir = basedir / "analysis"
    # outdir.mkdir(exist_ok=True, parents=True)

    assert isinstance(df.index, pd.MultiIndex), f"The input DataFrame index must be a MultiIndex, was {type(df.index)}"
    assert all([l in df.index.names for l in model_ids_by]), \
        f"The input DataFrame index must include the levels {model_ids_by}, but had the levels {df.index.names}."

    train_loss: pd.DataFrame = df[("train", "loss")].groupby(model_ids_by).agg("max").to_frame("train-loss")
    test_loss: pd.DataFrame = df[("test", "loss")].groupby(model_ids_by).agg("max").to_frame("test-loss")
    if include_validation:
        valid_loss: pd.DataFrame = df[("valid", "loss")].groupby(model_ids_by).agg("max").to_frame("valid-loss")
        loss_df = test_loss.join([train_loss, valid_loss])
    else:
        loss_df = test_loss.join([train_loss])

    return loss_df

    # if "idx" in df.index.names:
    #     df_no_idx = df.xs(0, level="idx")
    # else:
    #     df_no_idx = df
    #
    # fidelity_group = df_no_idx.groupby(level=fidelity_params)
    # counts = fidelity_group.size()
    # if save_tables:
    #     table = counts.unstack(level=fidelity_params[-1]).to_latex(caption="Fidelity-wise sample count.")
    #     with open(outdir / "fidelity_count_table.tex", "w") as fp:
    #         fp.write(table)
    # else:
    #     print(f"Fidelity-wise sample count:\n{counts.unstack(level=fidelity_params[-1])}")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(16, 9)) if save_plots else plt.subplots(1, 1)
    # fig: plt.Figure
    # ax: plt.Axes
    #
    # bar_width = 1
    # bin_loc_offset = 1
    # xticks = [[], []]
    # labels_to_colors = {}
    # all_colors = iter(plt.cm.Set2.colors)
    #
    # for level_0_key in counts.index.unique(0):
    #     subseries = counts.xs(level_0_key, level=0)
    #     ser_colors = [map_label_to_color(i, labels_to_colors, all_colors) for i in subseries.keys()]
    #     bin_locs = list(range(bin_loc_offset, bin_loc_offset + subseries.size, bar_width))
    #     ax.bar(bin_locs, height=subseries.values, width=bar_width, color=ser_colors)
    #     bin_loc_offset += subseries.size + 2
    #     xticks[0] += [sum(bin_locs) / subseries.size, ]
    #     xticks[1] += [level_0_key, ]
    #
    # ax.xaxis.set_ticks(xticks[0])
    # ax.xaxis.set_ticklabels(xticks[1])
    #
    # legend_labels = [f"{counts.index.names[1]}={k}" for k in labels_to_colors.keys()]
    # legend_lines = [Line2D([0], [0], linewidth=8, color=labels_to_colors[l]) for l in labels_to_colors.keys()]
    # ax.legend(legend_lines, legend_labels, loc="upper right", fontsize=16)
    # ax.set_title("Number of samples drawn from each fidelity level.")
    # ax.set_xlabel("Number of cell-repetitions per architecture (N).")
    # ax.set_ylabel("Number of samples drawn from the search space.")
    # ax.yaxis.set_tick_params(which='minor')
    # # ax.yaxis.grid(True, which='major')
    # # ax.yaxis.grid(True, which='minor', linestyle="--", linewidth=2., color='gray')
    # if save_plots:
    #     fig.tight_layout()
    #     fig.savefig(outdir / "per_fidelity_sample_counts.pdf")
    # else:
    #     plt.show()
    #
    # ### Analyze average model training time ###
    # mean_train_times = df["train_time"].groupby(level=df.index.names.difference(["idx"])).mean()
    # mean_train_times: pd.Series = mean_train_times.groupby(level=fidelity_params).mean()
    #
    # if save_tables:
    #     table = mean_train_times.to_frame().unstack(fidelity_params[-1]).to_latex(
    #         caption="Average training time per epoch across fidelity values.")
    #     with open(outdir / "per_fidelity_training_time.tex") as fp:
    #         fp.write(table)
    # else:
    #     print(f"Fidelity-wise average model training time per epoch:\n"
    #           f"{mean_train_times.to_frame().unstack(fidelity_params[-1])}")

    # fig, ax = plt.subplots(1, 1, figsize=(16, 9)) if save_plots else plt.subplots(1, 1)
    # fig: plt.Figure
    # ax: plt.Axes


# @df_loader_wrapper
# def group(basedir: Path = None, df: pd.DataFrame = None, groupby: list = None, columns: list = None):
#     """ Convenience function for generating specific groupings. The list "groupby" refers to column names that should
#     be used to group the data by and the list "columns" refers to those columns that should be present in the grouped
#     data. If "groupby" is None, the dataframe is returned unchanged. If "columns" is None, all columns other than those
#     in "groupby" are present in the returned group. """
#
#     if groupby is None:
#         return df, df
#
#     assert all([g in df.columns.names for g in groupby]), \
#         f"Mismatch between provided grouping parameters, {groupby}, and the dataframe's column names {df.columns.names}"
#
#     g = df.groupby(groupby)
#
#     pass


def collapse_index_names(index: pd.MultiIndex, levels: list = None, nlevels: int = None, separator: str = "-") -> \
        Union[pd.Index, pd.MultiIndex]:
    """ Collapses the index labels in a MultiIndex. The levels to be collapsed can be passed as either a list of
    specific levels in "levels" or an integer can be passed to "nlevels" indicating that the first "nlevels" levels
    should be collapsed into one level - the first level. """

    if levels is not None and nlevels is not None:
        raise RuntimeError("Either a list of index names or the number of indices to be collapsed should be provided, "
                           "not both.")

    if levels is not None:
        inds = [index.names.index(i) for i in levels]
        old_names = levels
    else:
        inds = list(range(nlevels))
        old_names = [index.names[i] for i in inds]

    compressor = separator.join([f"{{0[{i}]}}" for i in inds]).format

    if len(inds) > index.nlevels:
        raise RuntimeError(f"Number of levels to be collapsed - {len(inds)} - cannot exceed the number of levels in "
                           f"the original index - {index.nlevels}")
    if len(inds) < index.nlevels:
        remainder = [i for i in range(index.nlevels) if i not in inds]
        new_ind: list = [index.map(lambda i: (compressor(i)))] + [index.get_level_values(j) for j in remainder]
        new_names = [compressor(index.names)] + index.names.difference(old_names)
        new_ind: pd.MultiIndex = pd.MultiIndex.from_arrays(new_ind, names=new_names)
    else:
        new_ind: pd.Index = index.map(lambda i: compressor(i))
        new_ind.name = compressor(index.names)

    return new_ind


@_df_loader_wrapper
def rank_by_parameter(basedir: Path, df: pd.DataFrame, rank_on: Any, parameters: list = None, ascending=False,
                      **kwargs) -> pd.DataFrame:
    """ Generate ranks for the Dataframe's rows based on the column specified in 'rank_on'. If a list of additional
    column names is provided, a dataframe containing mean ranks of those parameters is returned, otherwise a copy of
    the initial dataframe with the "ranks" column tacked at the end is returned. """

    ranks = df[rank_on].rank(ascending=ascending, **kwargs).to_frame("rank")
    ranks = df.join([ranks])

    if parameters is not None:
        ranks = ranks.groupby(parameters)["rank"].agg("mean").to_frame("Rank")

    return ranks


@_df_loader_wrapper
def get_nsamples(basedir: Path, df: pd.DataFrame, groupby: list, index: Optional[list] = None, **kwargs) \
        -> pd.DataFrame:
    """
    Returns a DataFrame with a single column - "nsamples" - that contains the number of individual models (as
    identified by the tuple (taskid, model_idx)) that were observed. 'groupby' can be a list of column names that
    dictate the index of the returned DataFrame, i.e. the number of models in each group as identified by a unique
    combination of the respective values in the columns named in 'groupby' is counted. It is assumed that the input
    DataFrame's index does not need to be filtered anymore, e.g. for specific epochs. If it contains epoch-wise data,
    then "nsamples" will count each epoch as a separate data point.

    :param basedir:
    :param df:
    :param groupby: list
        A list of column names, compatible with df.groupby().
    :param index: list of strings
        The new index names - should be a one-to-one renaming of "groupby".
    :param kwargs:
    :return:
    """

    assert all([g in df.columns for g in groupby]), "Mismatch in input DataFrame's columns and given grouping parameters.\nGiven " \
                                  f"parameters:\n{groupby}\n\nDataFrame columns:\n{df.columns}"

    available_cols = df.columns.difference(groupby)
    nsamples = df.xs(1, level=MetricDFIndexLevels.epoch.value)
    nsamples = nsamples[[available_cols[0], *groupby]].groupby(groupby).agg("count")

    if isinstance(nsamples, pd.Series):
        nsamples = nsamples.to_frame("nsamples")
    else:
        assert isinstance(nsamples, pd.DataFrame), f"Expected a pandas DataFrame, but 'nsamples' is {type(nsamples)}"
        assert len(nsamples.columns) == 1, f"Expected a DataFrame with a single column, but the given DataFrame has " \
                                           f"{len(nsamples.columns)} columns."
        nsamples.columns = pd.Index(["nsamples"])

    # Reduce the names of the index levels to the bare minimum necessary
    assert len(index) == len(nsamples.index.names), \
        f"Mismatch in number of index names given - {len(index)} - and the number of available index levels - " \
        f"{nsamples.index.nlevels}"

    nsamples.index = nsamples.index.set_names(index)

    return nsamples


@_df_loader_wrapper
def estimate_remaining_runtime(basedir: Path, df: pd.DataFrame, max_epochs: int = 200) -> pd.DataFrame:
    """
    Given a complete metrics DataFrame, returns a DataFrame containing the estimated runtime needed to finish
    evaluating each model.
    :param df: pandas DataFrame
        A DataFrame object that can be read by the postprocessing modules to extract the number of epochs that each
        model has been evaluated for as well as how long each model has been run for.
    :param max_epochs:
    :return:
    """

    nepochs: pd.DataFrame = get_nepochs(df=df)
    remaining_epochs: pd.DataFrame = nepochs.rsub(max_epochs)

    runtimes: pd.Series = get_runtimes(df=df, reduce_epochs=True)["runtime"]
    runtime_per_epoch: pd.Series = runtimes / nepochs["nepochs"]
    required_runtimes: pd.DataFrame = (remaining_epochs["nepochs"] * runtime_per_epoch).to_frame("required")

    return required_runtimes


@_df_loader_wrapper
def get_op_counts(basedir: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a complete metrics DataFrame, returns a DataFrame containing the number of times each known operation type
    was used in each model configuration's cell architecture.
    :param df: pandas DataFrame
        A DataFrame object that can be read by the postprocessing modules to extract the number of epochs that each
        model has been evaluated for as well as how long each model has been run for.
    :return:
    """

    configs = get_configs(df=df)
    n_ops = len(EDGE_LIST)
    ops = configs[[f"Op{o+1}" for o in range(n_ops)]]
    max_cols = pd.Int64Index(list(range(len(OP_NAMES))))
    dummies = [pd.get_dummies(ops[c]) for c in ops.columns]
    for d in dummies:
        missing_cols = max_cols.difference(d.columns)
        d[missing_cols] = 0

    freq = sum(dummies)
    freq.columns = freq.columns.map(lambda x: OP_NAMES[x])

    assert (freq.sum(axis=1) == n_ops).all(), "Failed to verify integrity of cell operation frequency calculation."

    return freq


if __name__ == "__main__":
    args = _parse_cli()
    basedir: Path = args.basedir
    get_accuracies(basedir=basedir)

