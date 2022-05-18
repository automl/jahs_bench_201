import functools
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable
import yacs.config as config

from tabular_sampling.surrogate import loss, constants

_log = logging.getLogger(__name__)

def custom_loss_function(loss_params: config.CfgNode) -> Callable:
    """ Given a configuration, construct a custom loss function. """

    func_map = {
        constants.default_base_loss_configs.exp_lower_bound.name: loss.exponential_bounds,
        constants.default_base_loss_configs.exp_upper_bound.name: loss.exponential_bounds,
        constants.default_base_loss_configs.se.name: loss.squared_error
    }

    funcs = []
    weights = []
    for f, p, w in zip(loss_params.funcs, loss_params.params, loss_params.weights):
        if f == "":
            continue
        funcs.append(functools.partial(func_map[f], **p))
        weights.append(w)

    loss_func = loss.mix_objectives(*funcs, weights=weights)
    return loss_func

def load_pipeline_config(*params) -> config.CfgNode:
    """ Given a list of parameters, interprets the list as (key, value) pairs and
    converts them into an appropriate pipeline configuration. """

    pipeline_config = constants.default_pipeline_config.clone()
    pipeline_config.merge_from_list(params)

    try:
        loss_type = constants.RegressionLossFuncTypes(pipeline_config.loss)
    except ValueError as e:
        raise ValueError(
            f"The loss function for the pipeline should be a right-hand string from the "
            f"following mapping:\n{list(constants.RegressionLossFuncTypes)}") from e

    if loss_type is constants.RegressionLossFuncTypes.squared_error:
        return pipeline_config

    # Assume it's a custom loss function
    try:
        for f in pipeline_config.loss_params.funcs:
            if f == "":
                continue
            _ =  constants.default_base_loss_configs(f)
    except ValueError as e:
        raise ValueError(
            f"Unrecognized base loss function {f}, must be either '' or one of "
            f"{list(constants.default_base_loss_configs.__members__.keys())}") from e

    loss_params = pipeline_config.loss_params
    for i, p in enumerate(loss_params.params):
        # The parameters may be specified by passing in a filename instead
        if isinstance(p, str):
            cfg_file = Path(pipeline_config.config_dir) / p
            loss_params.params[i] = config.CfgNode().merge_from_file(cfg_file)

    return pipeline_config
