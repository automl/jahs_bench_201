import dataclasses as DC
from enum import Enum
from functools import partial
from typing import Union, Sequence

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, \
    mean_absolute_percentage_error, median_absolute_error

import logging
_log = logging.getLogger(__name__)

# Some scaffolding for handling fidelity values semantically and fluidly
@DC.dataclass(init=True, eq=True, frozen=True, repr=True, order=False, unsafe_hash=True)
class Fidelity:
    Resolution: Union[Sequence[int], slice] = DC.field(default_factory=list)
    W: Union[Sequence[int], slice] = DC.field(default_factory=list)
    N: Union[Sequence[int], slice] = DC.field(default_factory=list)
    Epoch: Union[Sequence[int], slice] = DC.field(default_factory=list)

def extract_relevant_indices(data: pd.DataFrame, fidelity: Fidelity) -> Union[pd.Index, pd.MultiIndex]:
    """ Given a dataframe in wide-format and a set of fidelity values as an instance of Fidelity, return a
    corresponding index which can be used to select rows corresponding to the given Fidelity instance. """
    # Extract the indices of only the relevant data
    x: pd.DataFrame = data
    x = x.loc[x[fids].isin(DC.asdict(fidelity)).all(axis=1)]
    relevant_index = x.index
    return relevant_index


class LongFormatIndexNames(Enum):
    SAMPLE_IDX = "Sample ID"
    VALUE_TYPE = "Value Type"

class LongFormatOutputTypes(Enum):
    TRUE = "True"
    PREDICTED = "Predicted"

class LongFormatValueTypes(Enum):
    FEATURE = "Feature"
    GROUP = "Group"
    TRUE = LongFormatOutputTypes.TRUE.value
    PREDICTED = LongFormatOutputTypes.PREDICTED.value


def wide_to_long_format(data: pd.DataFrame, output_type: LongFormatOutputTypes) \
        -> pd.DataFrame:
    """ Given a metrics dataframe in wide format, as used by the surrogate, returns an
    equivalent DataFrame in long-format. The parameter `output_type` should be set to one
    of LongFormatOutputTypes.TRUE or LongFormatOutputTypes.PREDICTED to indicate whether
    the values contained within correspond to metrics recorded in the performance
    dataset or were predicted by a surrogate model. The final long-format DataFrame
    treats features and metrics as values, one in each row, and indxes them using a
    combination of the integer index of each row in the wide-format DataFrame, called
    "Sample ID", and the type of value, which can be one of LongFormatValueTypes.FEATURE,
    LongFormatValueTypes.GROUP, LongFormatValueTypes.TRUE or
    LongFormatValueTypes.PREDICTED. LongFormatValueTypes.GROUP corresponds to the values
    of the columns "model_ID" and "fidelity_ID", defining groups of rows belonging to the
    same configuration and fidelity group, respectively.
    """

    _log.debug(f"Attempting to convert input wide-format DataFrame of shape {data.shape} "
               f"to long-format.")
    assert isinstance(output_type, LongFormatOutputTypes), \
        f"The parameter 'output_type' must be a member of the Enum " \
        f"LongFormatOutputTypes, was '{output_type}' of type '{type(output_type)}' " \
        f"instead."

    # Redundant, but ensures semantic compatibility
    output_type = LongFormatValueTypes(output_type.value)

    _log.debug("Splitting wide-format DataFrame's top-level columns into individual "
               "frames.")
    index = data.index.to_frame(index=True, name=LongFormatIndexNames.SAMPLE_IDX.value)

    groups: pd.DataFrame = data.loc[:, "sampling_index"]
    groups: pd.DataFrame = groups.loc[:, ["fidelity_ID", "model_ID"]]
    groups = groups.assign(
        **{LongFormatIndexNames.VALUE_TYPE.value: LongFormatValueTypes.GROUP.value})
    groups = pd.concat([index, groups], axis=1)

    features: pd.DataFrame = data.loc[:, "features"]
    features = features.assign(
        **{LongFormatIndexNames.VALUE_TYPE.value: LongFormatValueTypes.FEATURE.value})
    features = pd.concat([index, features], axis=1)

    labels: pd.DataFrame = data.loc[:, "labels"]
    labels = labels.assign(**{LongFormatIndexNames.VALUE_TYPE.value: output_type.value})
    labels = pd.concat([index, labels], axis=1)

    id_vars = [v.value for v in LongFormatIndexNames]

    _log.debug(f"Performing sanity checks on component frames.")
    frames = {"groups": groups, "features": features, "labels": labels}
    for f1 in frames.items():
        _log.debug(f"LHS frame: {f1[0]}\nColumns: {f1[1].columns}")
        for f2 in frames.items():
            if f1[0] == f2[0]:
                continue
            _log.debug(f"RHS frame: {f2[0]}")
            overlap = f1[1].columns.intersection(f2[1].columns)
            overlap = overlap.difference(id_vars)
            if overlap.size != 0:
                raise RuntimeError(
                    f"Failed to create long-format DataFrame due to the presence of "
                    f"overlapping column names {overlap.tolist()} between '{f1[0]}' and "
                    f"'{f2[0]}'."
                )
    _log.debug("Sanity check successfully passed, melting DataFrame.")

    groups_long = pd.melt(groups, id_vars=id_vars, var_name="variable",
                          value_name="value", ignore_index=True)
    features_long = pd.melt(features, id_vars=id_vars, var_name="variable",
                            value_name="value", ignore_index=True)
    labels_long = pd.melt(labels, id_vars=id_vars, var_name="variable",
                          value_name="value", ignore_index=True)

    long_data = pd.concat([groups_long, features_long, labels_long], axis=0)
    return long_data


def generate_ranks_dataframe(outputs: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """ Given a DataFrame containing the values for any number of outputs in long-format, where all output
    values are under a single column 'value', return a similar DataFrame where the 'value' column's
    contents are replaced by their respective ranks within the group defined by the columns in 'group_cols'. """

    x = outputs.copy()
    g = x.groupby(group_cols)
    ranks = g["value"].rank()
    x.loc[:, "value"] = ranks["value"]
    return x


def generate_correlations_df(values, index: pd.MultiIndex) -> pd.DataFrame:
    """ Performs a bunch of pandas acrobatics in order to convert the given numpy 2D Array 'values' into a long
    format DataFrame using the given heirarchical index. The index names are directly used as identity variables
    and the prefix "val-" is appended to them for the value variables. """

    columns = index.rename([f"val-{n}" for n in index.names])
    x = pd.DataFrame(values, index=index, columns=columns)
    x = x.stack(columns.names).to_frame("value")
    x = x.reset_index()
    return x


def generate_correlations(ranks: pd.DataFrame, group_cols: Sequence[str] = None) -> pd.DataFrame:
    """ Given a long-format DataFrame containing the rankings across the columns specified in 'group_cols' and
    of any number of different outputs (specified under the column 'output_var') in a column named 'value',
    calculates their Spearman and Kendall-Tau rank correlation values and p-values and arranges them in a tidy
    long format DataFrame. The output DataFrame is structured such that the identification variables
    'Metric Type' and 'Value Type' are added to  the input 'ranks' DataFrame's structure, where 'Metric Type'
    can be either 'Spearman' or 'Kendall-Tau'  whereas 'Value Type' can be either 'Correlation' or 'P-Value'.
    Correlations are calculated across the rankings in the same group. """

    x = ranks.copy()
    intra_group_indices = x.groupby(group_cols)["value"].cumcount()
    x.loc[:, "igidx"] = intra_group_indices.values
    x = x.set_index(group_cols + ["igidx"])[["value"]]
    x = x.unstack(group_cols)

    new_index = x.columns.get_loc_level("value")[1]  # Remove unused column

    # Calculate Spearman's R-value for rank correlations
    corr, pval = spearmanr(x.values, nan_policy="omit")
    corrdf_sp = generate_correlations_df(corr, index=new_index).assign(
        **{"Metric Name": "Spearman", "Value Type": "Correlation"})
    pvaldf_sp = generate_correlations_df(pval, index=new_index).assign(
        **{"Metric Name": "Spearman", "Value Type": "p-Value"})

    # Calculate Kendall-Tau value for rank correlations
    corr, pval = spearmanr(x.values, nan_policy="omit")
    corrdf_kt = generate_correlations_df(corr, index=new_index).assign(
        **{"Metric Name": "Kendall-Tau", "Value Type": "Correlation"})
    pvaldf_kt = generate_correlations_df(pval, index=new_index).assign(
        **{"Metric Name": "Kendall-Tau", "Value Type": "p-Value"})

    final_df = pd.concat([corrdf_sp, pvaldf_sp, corrdf_kt, pvaldf_kt], axis=0)
    return final_df


def map_index(index: pd.MultiIndex, mapping: dict, level: int = 0) -> pd.MultiIndex:
    """ Simple functionality that pandas itself really should have. Applies a mapping to only select elements of a
    given MultiIndex and returns the new MultiIndex. """

    index_level_values = index.to_frame(index=False)
    map_fn = lambda y: mapping[y] if y in mapping else y
    index_level_values[level] = list(map(map_fn, index_level_values[level]))
    return pd.MultiIndex.from_frame(index_level_values)


def get_filtered_data(data: pd.DataFrame, relevant_fidelity: Fidelity, relevant_outputs: Sequence[str] = None,
                      adjust_for_minimization: bool = True) -> \
        (pd.DataFrame, Sequence[str]):
    """ Filter out the full dataset according to the given set of fidelities and the required outputs, returning a
    proper long-format DataFrame. Also returns the DataFrame structure in the form of a list containing the
    identifier variables, which includes the list of input features (configuration) and 'output_type', which
    indicates whether a value belongs to the set of actual observations (output_type="True") or the set of
    surrogate predictions (output_type="Predicted"). The value variables are always placed in 'output_vars' and the
    actual values are always placed in 'value'. The flag 'adjust_for_minimization', when True (default), adjusts a
    number of metrics such that they are suitable for use in minimization. This includes all accuracy metrics,
    identified by the suffix '-acc' and the metric FLOPS. Accuracy metrics are relabeled to '-err' to indicate
    Error% instead of Accuracy%. """

    # Handle the case-sensitivity issues with the "Epoch" parameter
    mapping = {"epoch": "Epoch"}
    data.columns = map_index(data.columns, mapping, level=1)

    # Make sure that it is always possible to numerically identify which samples belong together
    data = data.rename_axis(["Sample ID"], axis=0).reset_index(col_fill="sampling_index",
                                                               col_level=1)
    relevant_index = extract_relevant_indices(data.features, relevant_fidelity)
    relevant_data = data.loc[relevant_index].copy()

    rename_accs = {}
    if adjust_for_minimization:
        # Convert accuracy to error in order to enable all outputs to have the same ordering i.e. a smaller value
        # is better
        acc_cols = [c for c in relevant_data.columns if "-acc" in c[1]]
        relevant_data.loc[:, acc_cols] = relevant_data.loc[:, acc_cols].rsub(100).values
        rename_accs = {c[1]: c[1].replace("-acc", "-err") for c in acc_cols}

        flops_cols = [c for c in relevant_data.columns if "FLOPS" in c[1]]
        relevant_data.loc[:, flops_cols] = relevant_data.loc[:, flops_cols].mul(-1).values

    # Re-scale size_MB to size_KB in order to make plotting easier
    size_cols = [c for c in relevant_data.columns if "size_MB" in c[1]]
    relevant_data.loc[:, size_cols] = relevant_data.loc[:, size_cols].mul(1000).values
    rename_sizes = {c[1]: c[1].replace("_MB", "_KB") for c in size_cols}

    # Filter out the outputs if needed
    if relevant_outputs is not None:
        drop_cols = relevant_data.labels.columns.difference(relevant_outputs)
        relevant_data = relevant_data.drop([("labels", c) for c in drop_cols], axis=1)

    rename_map = {**rename_accs, **rename_sizes}
    relevant_data.columns = map_index(relevant_data.columns, rename_map, level=1)
    return relevant_data


def get_correlations(data: pd.DataFrame, extra_group_by: Sequence[str] = None) -> pd.DataFrame:
    group_cols = ["output_type", "output_var"] + ([] if extra_group_by is None else extra_group_by)
    all_ranks = generate_ranks_dataframe(data, group_cols=group_cols)
    correlations = generate_correlations(all_ranks, group_cols=group_cols)
    return correlations


# TODO: Re-define this in order to change the default arguments, e.g. to change 'multioutput'
score_funcs = {
    "R2": partial(r2_score, multioutput="raw_values"),
    "RMSE": partial(mean_squared_error, multioutput="raw_values", squared=False),
    "MAPE": partial(mean_absolute_percentage_error, multioutput="raw_values"),
    "MedAE": partial(median_absolute_error, multioutput="raw_values"),
}


def get_scores(data: pd.DataFrame, group_cols: Sequence[str] = None):
    """ Given a long-format dataframe of benchmark data, with all value variables under the column 'output_var',
    the label for "True" vs "Predicted" values under the column 'output_type' and all output values under the
    column 'value', return a long-format DataFrame with the regression analysis scores of the given dataset
    such that the column 'value' now refers to a metric's score, the metric name is stored in the column
    'metric_type' and the other columns serve as identifier variables corresponding to the column 'output_var'
    from the original data and any other grouping columns specified in 'group_cols'. """

    group_cols = [] if group_cols is None else group_cols
    x = data.set_index(data.columns.difference(["value"]).to_list())
    g = x.unstack(["output_type"]).droplevel(0, axis=1).groupby(["output_var"] + group_cols)

    scores = {k: g.apply(lambda x: f(x.loc[:, "True"], x.loc[:, "Predicted"])) for k, f in score_funcs.items()}
    scores = pd.DataFrame(scores, columns=pd.Index(list(scores.keys()), name="metric_type"))
    scores = scores.explode(list(scores.keys()))
    scores = scores.stack(["metric_type"]).rename("value").reset_index()
    return scores

    # TODO: Test for multiple outputs


def get_test_frac_data(test_frac: float, filter_loss: bool = True):
    """ Load up the data for the model trained using a particular dataset split, specified by the float
    'test_frac', from disk, filter and clean the data and return it as a wide-format dataframe with
    2-level MultiIndex columns. """
    datadir = basedir / f"test-frac-{test_frac}"
    test_set_pth = datadir / "test_set.pkl.gz"
    test_pred_pth = datadir / "test_pred.pkl.gz"
    test_set = pd.read_pickle(test_set_pth)
    ypred = pd.read_pickle(test_pred_pth)
    test_set.rename({"labels": "true"}, axis=1, inplace=True)
    test_set.rename({"epoch": "Epoch"}, axis=1, inplace=True)
    test_set.drop([c for c in test_set.columns if "-loss" in c[1]], axis=1, inplace=True)
    ypred.drop([c for c in ypred.columns if "-loss" in c[1]], axis=1, inplace=True)
    ypred = ypred.droplevel(0, axis=1)
    ypred.columns = pd.MultiIndex.from_product([["predicted"], ypred.columns.to_list()])
    assert ypred.shape[0] == test_set.shape[0]
    ypred.index = test_set.index
    data = pd.concat([test_set, ypred], axis=1)
    return data

def get_filtered_corr(data: pd.DataFrame, extra_group_by: Sequence[str] = None):
    correlations = get_correlations(data=data, extra_group_by=extra_group_by)
    x = correlations
    x = x.loc[x.output_type=="Predicted"]
    x = x.loc[x["val-output_type"]=="True"]
    x = x.loc[x["Metric Name"]=="Kendall-Tau"]
    return x[["Value Type", "value"]]

# maxfidelity = Fidelity(N=[5], W=[16], Resolution=[1.0], Epoch=[200])
# maxcapacity = Fidelity(N=[5], W=[16], Resolution=[1.0], Epoch=list(range(1, 201)))
# maxbudget = Fidelity(N=[1, 3, 5], W=[4, 8, 16], Resolution=[0.25, 0.5, 1.0], Epoch=[200])
fids = ["Resolution", "W", "N", "Epoch"]

