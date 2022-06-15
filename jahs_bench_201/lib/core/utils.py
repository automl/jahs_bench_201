from typing import Dict, Union, Any, Optional, List, Tuple

import ConfigSpace
import pandas as pd


# TODO: Update documentation
def adapt_search_space(
    original_space: Union[ConfigSpace.ConfigurationSpace, Any],
    portfolio: Optional[pd.DataFrame] = None, taskid: Optional[int] = None,
    opts: Optional[Union[Dict[str, Any], List[str]]] = None,
    suffix: Optional[str] = "_custom") -> Union[
        Tuple[ConfigSpace.ConfigurationSpace, bool], bool]:
    """ Given a ConfigurationSpace object and a valid configuration, restricts the respective configuration space
    object and consequently the overall search space by setting the corresponding parameters to constant values. Such
    a configuration may be provided either by means of a portfolio file along with the relevant taskid or a dictionary
    'opts' mapping strings to values of any type, or both in which case values in 'opts' take precedence. In both
    cases, the configuration is read as a dictionary and the function attempts to autonomously restrict the search
    space such that for each key *k*, the corresponding parameter is set to the constant value *v*. An optional
    string 'suffix' may be provided to modify the name of the ConfigSpace object. Default: '_custom'. If None, the
    name remains unchanged. The name also remains unchanged in the case when the given space was not modified at all,
    most likely because no valid keys were present in the configuration dictionary or the dictionary was empty. In
    such a case, False is returned. Otherwise, True is returned. """

    new_consts = {}
    if portfolio is None and opts is None:
        return False
    elif portfolio is not None:
        if taskid is None:
            raise ValueError(
                f"When a portfolio is given, an integer taskid must also be provided. Was given {taskid}")
        else:
            new_consts = portfolio.loc[taskid % portfolio.index.size, :].to_dict()

    # Convert opts from list of string pairs to a mapping from string to string.
    if isinstance(opts, List):
        i = iter(opts)
        opts = {k: v for k, v in zip(i, i)}
    new_consts = {**new_consts, **opts}

    if hasattr(original_space, "config_space"):
        config_space = original_space.config_space
        flag_cs_attr = True
    else:
        config_space = original_space
        flag_cs_attr = False

    known_params = {p.name: p for p in config_space.get_hyperparameters()}

    def param_interpretor(param, value):
        known_config_space_value_types = {
            ConfigSpace.UniformIntegerHyperparameter: int,
            ConfigSpace.UniformFloatHyperparameter: float,
            ConfigSpace.CategoricalHyperparameter: lambda x: type(param.choices[0])(x),
            ConfigSpace.OrdinalHyperparameter: lambda x: type(param.sequence[0])(x),
        }
        return known_config_space_value_types[type(param)](value)

    modified = False

    for arg, val in new_consts.items():
        if arg in known_params:
            old_param = known_params[arg]
            new_val = param_interpretor(old_param, val)
            if isinstance(new_val, bool):
                new_val = int(
                    new_val)  # Because ConfigSpace doesn't allow Boolean constants.
            known_params[arg] = ConfigSpace.Constant(arg, new_val,
                                                     meta=dict(old_param.meta, **dict(
                                                         constant_overwrite=True)))
            modified = True

    if modified:
        new_config_space = ConfigSpace.ConfigurationSpace(
            f"{config_space.name}{suffix if suffix is not None else ''}")
        new_config_space.add_hyperparameters(known_params.values())
        if flag_cs_attr:
            original_space.config_space = new_config_space
        else:
            return new_config_space, modified

    return modified
