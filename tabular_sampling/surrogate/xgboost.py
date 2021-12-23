import functools
import pandas as pd
import xgboost as xgb
from tabular_sampling.search_space.configspace import joint_config_space
import ConfigSpace as cs
from typing import Union, Optional, Iterable, Sequence, Callable, List, Any
import typing_extensions as typingx
import logging

_log = logging.getLogger(__name__)
ConfigType = Union[dict, cs.Configuration]

def get_one_hot_encoder(parameter: cs.CategoricalHyperparameter) -> Callable:
    """ Generate a callable function tailored for the given categorical hyper-parameter, which, when given a valid
    value for the hyper-parameter generates a one-hot encoding of that value as a dictionary. """

    codes = {f"{parameter.name}_{c}": 0 for c in parameter.choices}
    def encoder(pval: Union[pd.Series, Any]):
        if isinstance(pval, pd.Series):
            assert pval.isin(parameter.choices).all(), \
                f"Unknown values {pval[pval.isin(parameter.choices) == False].unique()} for parameter {parameter.name}."

            # Get the one-hot encoding
            encoding = pd.get_dummies(pval, prefix=parameter.name, prefix_sep="_")

            # Ensure some consistency
            expected_cols = pd.Index(codes.keys())
            missing = expected_cols.difference(encoding.columns)
            encoding[missing] = 0  # Fill in missing columns
            encoding = encoding[expected_cols]  # Re-order the columns
            return encoding
        else:
            assert pval in parameter.choices, f"Unknown value {pval} for parameter {parameter.name}."
            encoding = dict(**codes)
            encoding[f"{parameter.name}_{pval}"] = 1
            return encoding

    return encoder


def get_identity_encoder(parameter: cs.hyperparameters.Hyperparameter) -> Callable:
    """ Generates an identity encoder that does not change the input parameter value at all. Used for the sake of
    consistency across vector operations. """

    def identity(pval: Union[pd.Series, Any]):
        return pval if isinstance(pval, pd.Series) else {parameter.name: pval}

    return identity


def get_default_encoders(space: cs.ConfigurationSpace) -> dict:
    """ For a given configuration space, returns a dict mapping each parameter name in the space to an encoder for that
    parameter's type. This is the default set of encoders. """

    params = space.get_hyperparameters()

    identity_types = [cs.OrdinalHyperparameter, cs.UniformFloatHyperparameter, cs.UniformIntegerHyperparameter]
    special_types = {cs.CategoricalHyperparameter: get_one_hot_encoder}

    encoder_by_type = {
        **{p: get_identity_encoder for p in identity_types},
        **{p: enc for p, enc in special_types.items()}
    }

    encoder_by_name = {p.name: encoder_by_type[type(p)](p) for p in params}
    return encoder_by_name


class XGBoost:
    """ A surrogate model based on XGBoost. """

    config_space: cs.ConfigurationSpace
    encoders: dict
    _xheaders = None
    _trained = False

    def __init__(self, config_space: Optional[cs.ConfigurationSpace] = joint_config_space,
                 encoders: Optional[dict] = None):
        self.config_space = config_space
        self.encoders = encoders if encoders is not None else get_default_encoders(self.config_space)

        # Both initializes some internal attributes as well as performs a sanity test
        default_config = self.config_space.get_default_configuration()
        default_encoding = self.encode(default_config)
        pass

    def _encode_single(self, config: ConfigType) -> dict:
        """ Encode a single configuration. """

        if isinstance(config, cs.Configuration):
            config = config.get_dictionary()

        if isinstance(config, dict):
            encodings: Sequence[dict] = [self.encoders[p](v) for p, v in config.items()]
            # Concatenate all the encodings into one dict
            config: dict = functools.reduce(lambda c, e: {**c, **e}, encodings)
        else:
            TypeError(f"Unrecognized input type: {type(config)}. Must be {ConfigType}.")

        return config

    def _encode_series(self, param_values: pd.Series) -> pd.DataFrame:
        """ Encode a Pandas Series that describes multiple values of a single parameter type. The name of the Series
        'param_values' is treated as the parameter name. This can, thus, be directly passed to the method 'apply()' of
        a pandas.DataFrame instance in order to encode the entire DataFrame of complete configurations. Since different
        kinds of encodings can generate either a Series or a DataFrame object (e.g. one-hot encoding), any Series
        encodings are returned as DataFrame objects as well. Such DataFrames have a single column: the name of the
        Series. """

        encodings = self.encoders[param_values.name](param_values)
        if isinstance(encodings, pd.Series):
            encodings = encodings.to_frame(encodings.name)

        return encodings

    def encode(self, data: Union[ConfigType, List[ConfigType], pd.DataFrame]):
        """ Encode the given data, assumed to be either a single configuration or a number of configurations, according
        to the defined rules for this model's config space. When passing a DataFrame, care must be taken that each row
        corresponds to a single configuration and each column to a single known parameter name. """

        if isinstance(data, typingx.get_args(ConfigType)):
            encodings: dict = self._encode_single(data)
            xheaders = list(encodings.keys())

            # Ensure consistency of ordering
            if self._xheaders is not None and xheaders is not None and self._xheaders != xheaders:
                encodings = {h: encodings[h] for h in self._xheaders}

        elif isinstance(data, list):
            encodings = list(map(self._encode_single, data))
            xheaders = list(encodings[0].keys())

            # Ensure consistency of ordering
            if self._xheaders is not None and xheaders is not None and self._xheaders != xheaders:
                encodings = map(lambda enc: {h: enc[h] for h in self._xheaders}, encodings)

        elif isinstance(data, pd.DataFrame):
            encodings: pd.DataFrame = pd.concat([self._encode_series(data[c]) for c in data.columns], axis=1)
            xheaders = encodings.columns

            # Ensure consistency of ordering
            if self._xheaders is not None and xheaders is not None and not xheaders.equals(pd.Index(self._xheaders)):
                encodings = encodings[self._xheaders]
        else:
            raise TypeError(f"Unrecognized input type for encode: {type(data)}. Must be "
                            f"{Union[ConfigType, Sequence[ConfigType], pd.DataFrame]}.")

        if self._xheaders is None:
            self._xheaders = xheaders

        return encodings

    def train(self, X, y):
        """ Pre-process the given dataset and fit an XGBoost model on it. """

        # TODO: Pre-process data, fit model
        pass

    def predict(self, X):
        """ Given some input data, generate model predictions. """
        pass
