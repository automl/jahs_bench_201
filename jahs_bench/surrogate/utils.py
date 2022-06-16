import functools
import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import sklearn.base
import yacs.config as config

from jahs_bench.surrogate import config as surrogate_config
from jahs_bench.surrogate import loss

_log = logging.getLogger(__name__)


def custom_loss_function(loss_params: config.CfgNode) -> Callable:
    """ Given a configuration, construct a custom loss function. """

    func_map = {
        surrogate_config.default_base_loss_configs.exp_lower_bound.name: loss.exponential_bounds,
        surrogate_config.default_base_loss_configs.exp_upper_bound.name: loss.exponential_bounds,
        surrogate_config.default_base_loss_configs.se.name: loss.squared_error
    }

    funcs = []
    weights = []
    for f, p, w in zip(loss_params.funcs, loss_params.params, loss_params.weights):
        if not f:
            continue
        funcs.append(functools.partial(func_map[f], **p))
        weights.append(w)

    loss_func = loss.mix_objectives(*funcs, weights=weights)
    return loss_func


def load_pipeline_config(*params) -> config.CfgNode:
    """ Given a list of parameters, interprets the list as (key, value) pairs and
    converts them into an appropriate pipeline configuration. """

    pipeline_config = surrogate_config.default_pipeline_config.clone()
    pipeline_config.defrost()
    pipeline_config.set_new_allowed(True)
    pipeline_config.merge_from_list(params)

    try:
        loss_type = surrogate_config.RegressionLossFuncTypes(pipeline_config.loss)
    except ValueError as e:
        raise ValueError(
            f"The loss function for the pipeline should be a right-hand string from the "
            f"following mapping:\n{list(surrogate_config.RegressionLossFuncTypes)}") from e

    if loss_type is surrogate_config.RegressionLossFuncTypes.squared_error:
        return pipeline_config

    # Assume it's a custom loss function
    try:
        for f in pipeline_config.loss_params.funcs:
            if not f:
                continue
            _ = surrogate_config.default_base_loss_configs(f)
    except ValueError as e:
        raise ValueError(
            f"Unrecognized base loss function {f}, must be either '' or one of "
            f"{list(surrogate_config.default_base_loss_configs.__members__.keys())}") from e

    loss_params = pipeline_config.loss_params
    for i, p in enumerate(loss_params.params):
        # The parameters may be specified by passing in a filename instead
        if isinstance(p, str):
            cfg_file = Path(pipeline_config.config_dir) / p
            new_params = config.CfgNode(new_allowed=True).merge_from_file(cfg_file)
            loss_params.params[i] = new_params

    target_config = pipeline_config.target_config
    for i, p in enumerate(target_config.params):
        # The parameters may be specified by passing in a filename instead
        if isinstance(p, str):
            cfg_file = Path(pipeline_config.config_dir) / p
            new_params = config.CfgNode(new_allowed=True).merge_from_file(cfg_file)
            target_config.params = new_params

    pipeline_config.set_new_allowed(False)
    pipeline_config.freeze()
    return pipeline_config


class CustomTransformFunctions:
    @staticmethod
    def sigmoid(arr: np.array, k: float = 1.):
        if k <= 0:
            raise ValueError(f"The parameter 'k' for a sigmoid function must be a "
                             f"positive real number, was {k}.")
        return np.power(1 + np.exp(-k * arr), -1)

    @staticmethod
    def inverse_sigmoid(arr: np.array, k: float = 1.):
        if k <= 0:
            raise ValueError(f"The parameter 'k' for an inverse sigmoid function must be "
                             f"a positive real number, was {k}.")
        if not (arr.min() > 0. and arr.max() < 1.0):
            raise ValueError(f"The domain of the inverse sigmoid function is the open "
                             f"interval (0, 1), but the given inputs have span the "
                             f"closed interval [{arr.min()}, {arr.max()}].")
        return (np.log(arr) - np.log(1 - arr)) / k


# Valid transformations for the target variable
class TargetTransforms(Enum):
    MinMax = sklearn.preprocessing.MinMaxScaler


class CustomTransforms(Enum):
    sigmoid = (CustomTransformFunctions.sigmoid, CustomTransformFunctions.inverse_sigmoid)
    inverse_sigmoid = \
        (CustomTransformFunctions.inverse_sigmoid, CustomTransformFunctions.sigmoid)


def apply_target_transform(target_config: config.CfgNode, final_estimator) -> \
    sklearn.base.BaseEstimator:
    """ Given a configuration describing how the targets should be transformed and a
    corresponding estimator, returns a MetaEstimator containing a pipeline such that the
    required transformations are applied to the targets before passed to
    'final_estimator'. The 'target_config' may describe either a single transformation or
    a chain of transformations. """

    if target_config.transform.lower() == "chain":
        # Chain the transformations by "prepending" each transformation (starting from
        # the last one) to the original estimator. This is necessary because sklearn
        # Pipelines don't support target transformations.
        estimator = final_estimator
        funcs = target_config.params["funcs"][::-1]
        params = target_config.params["params"][::-1]
        for f, p in zip(funcs, params):
            transformer = _get_single_transform(f, **p)
            estimator = sklearn.compose.TransformedTargetRegressor(
                transformer=transformer, regressor=estimator)
    else:
        transformer = _get_single_transform(target_config.transform,
                                            **target_config.params)
        estimator = sklearn.compose.TransformedTargetRegressor(
            transformer=transformer, regressor=final_estimator)

    return estimator


def _get_single_transform(name: str, **params) -> sklearn.base.BaseEstimator:
    """ Given the name of a known Transform and the relevant parameters, initialize a
    corresponding Transformer object and return it. """

    if "custom" in name:
        # Construct a custom transformer
        fname = name.split(":")[1]
        func, inv = CustomTransforms[fname].value
        return sklearn.preprocessing.FunctionTransformer(
            func=func, inverse_func=inv, kw_args=params, inv_kw_args=params)
    else:
        # Use an existing Transformer
        return TargetTransforms[name].value(**params)


def adjust_dataset(datapth: Path, outputs: Optional[Sequence[str]] = None,
                   fillna: bool = False):
    _log.info(f"Adjusting data from {datapth}. "
              f"{(' Chosen output columns: ' + str(outputs)) * (outputs is not None)}. "
              f"{'Filling ' if fillna else 'Not filling '} in NaN values. ")
    _log.info("Fetching data.")
    data = pd.read_pickle(datapth)
    _log.info("Finished loading data.")

    _log.info(f"There are {data.index.size} rows of data, including the test set (if "
              f"any).")
    index = data.loc[:, "sampling_index"]
    groups = index["model_ID"]
    strata = index["fidelity_ID"]
    features = data.features
    labels: pd.DataFrame = data.labels

    if outputs is not None:
        labels: pd.DataFrame = labels[outputs]

    if fillna:
        sel: pd.Series = labels.isna().any(axis=1)
        subdf = labels.loc[sel]
        if subdf.shape[0] == 0:
            _log.info("Found no NaN values in the labels.")
        else:
            _log.info(f"Found {subdf.shape[0]} rows of data with NaN values in them. "
                      f"Attempting to fill NaN values.")
            fillvals = {}
            for c in sel.index:
                if sel[c]:
                    if "acc" in c:
                        fillvals[c] = 0.
                    elif "loss" in c or "duration" in c:
                        fillvals[c] = 100 * labels[c].max()
                    else:
                        raise RuntimeError(f"Could not find appropriate fill value to "
                                           f"replace NaNs with for metric: {c}")
            _log.info(f"Filling NaN values according to the mapping: {fillvals}")
            labels.fillna(fillvals, inplace=True)

    return features, labels, groups, strata
