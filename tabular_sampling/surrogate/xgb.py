import functools

import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing, sklearn.compose, sklearn.multioutput
import xgboost as xgb
from tabular_sampling.search_space.configspace import joint_config_space
import ConfigSpace as cs
from typing import Union, Optional, Iterable, Sequence, Callable, List, Any, Dict
import typing_extensions as typingx
import logging

_log = logging.getLogger(__name__)
ConfigType = Union[dict, cs.Configuration]


class XGBSurrogate:
    """ A surrogate model based on XGBoost. Adapted from naslib.predictors.tree.xgb.XGBoost at
    https://github.com/automl/NASLib/blob/6506a700f4d4764f9c58c918efb6769967b55669/naslib/predictors/trees/xgb.py """

    config_space: cs.ConfigurationSpace
    encoders: dict
    feature_headers: List[str]
    label_headers: List[str]
    hyperparams: dict
    estimators_per_output: int
    _trained = False

    _hpo_search_space = {
            "objective": ["reg:squarederror"],
            # "eval_metric": "rmse",
            # 'early_stopping_rounds': 100,
            "booster": ["gbtree"],
            "max_depth": np.arange(1, 15),
            "min_child_weight": list(range(1, 10)),
            "colsample_bytree": scipy.stats.uniform(0.0, 1.0),
            "learning_rate": scipy.stats.loguniform(0.001, 0.5),
            # 'alpha': 0.24167936088332426,
            # 'lambda': 31.393252465064943,
            "colsample_bylevel": scipy.stats.uniform(0.0, 1.0),
        }

    @property
    def default_hyperparams(self) -> dict:
        params = {
            "objective": "reg:squarederror",
            # "eval_metric": "rmse",
            "booster": "gbtree",
            "max_depth": 6,
            "min_child_weight": 1,
            "colsample_bytree": 1,
            "learning_rate": 0.3,
            "colsample_bylevel": 1,
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
                "objective": "reg:squarederror",
                # "eval_metric": "rmse",
                # 'early_stopping_rounds': 100,
                "booster": "gbtree",
                "max_depth": int(np.random.choice(range(1, 15))),
                "min_child_weight": int(np.random.choice(range(1, 10))),
                "colsample_bytree": np.random.uniform(0.0, 1.0),
                "learning_rate": loguniform(0.001, 0.5),
                # 'alpha': 0.24167936088332426,
                # 'lambda': 31.393252465064943,
                "colsample_bylevel": np.random.uniform(0.0, 1.0),
            }
        self.hyperparams = params
        return params

    def __init__(self, config_space: Optional[cs.ConfigurationSpace] = joint_config_space,
                 estimators_per_output: int = 500, encoders: Optional[dict] = None, use_gpu: Optional[bool] = None):
        self.config_space = config_space
        self.estimators_per_output = estimators_per_output
        self.hyperparams = None
        self.use_gpu = use_gpu

        # Both initializes some internal attributes as well as performs a sanity test
        self.set_random_hyperparams()  # Sets default hyperparameters

    # def _encode_single(self, config: ConfigType) -> dict:
    #     """ Encode a single configuration. """
    #
    #     if isinstance(config, cs.Configuration):
    #         config = config.get_dictionary()
    #
    #     if isinstance(config, dict):
    #         encodings: Sequence[dict] = [self.encoders[p](v) for p, v in config.items()]
    #         # Concatenate all the encodings into one dict
    #         config: dict = functools.reduce(lambda c, e: {**c, **e}, encodings)
    #     else:
    #         TypeError(f"Unrecognized input type: {type(config)}. Must be {ConfigType}.")
    #
    #     return config
    #
    # def _encode_series(self, param_values: pd.Series) -> pd.DataFrame:
    #     """ Encode a Pandas Series that describes multiple values of a single parameter type. The name of the Series
    #     'param_values' is treated as the parameter name. This can, thus, be directly passed to the method 'apply()' of
    #     a pandas.DataFrame instance in order to encode the entire DataFrame of complete configurations. Since different
    #     kinds of encodings can generate either a Series or a DataFrame object (e.g. one-hot encoding), any Series
    #     encodings are returned as DataFrame objects as well. Such DataFrames have a single column: the name of the
    #     Series. """
    #
    #     encodings = self.encoders[param_values.name](param_values)
    #     if isinstance(encodings, pd.Series):
    #         encodings = encodings.to_frame(encodings.name)
    #
    #     return encodings
    #
    # def encode(self, features: Union[ConfigType, List[ConfigType], pd.DataFrame]) -> \
    #         Union[dict, List[dict], pd.DataFrame]:
    #     """ Encode the given features data, assumed to be either a single configuration or a number of configurations,
    #     according to the defined rules for this model's config space. When passing a DataFrame, care must be taken that
    #     each row corresponds to a single configuration and each column to a single known parameter name. """
    #
    #     if isinstance(features, typingx.get_args(ConfigType)):
    #         encodings: dict = self._encode_single(features)
    #         xheaders = list(encodings.keys())
    #
    #         # Ensure consistency of ordering
    #         if self.feature_headers is not None and xheaders is not None and self.feature_headers != xheaders:
    #             encodings = {h: encodings[h] for h in self.feature_headers}
    #
    #     elif isinstance(features, list):
    #         encodings = list(map(self._encode_single, features))
    #         xheaders = list(encodings[0].keys())
    #
    #         # Ensure consistency of ordering
    #         if self.feature_headers is not None and xheaders is not None and self.feature_headers != xheaders:
    #             encodings = map(lambda enc: {h: enc[h] for h in self.feature_headers}, encodings)
    #
    #     elif isinstance(features, pd.DataFrame):
    #         encodings: pd.DataFrame = pd.concat([self._encode_series(features[c]) for c in features.columns], axis=1)
    #         xheaders = encodings.columns.tolist()
    #
    #         # Ensure consistency of ordering
    #         if self.feature_headers is not None and xheaders is not None and self.feature_headers != xheaders:
    #             encodings = encodings[self.feature_headers]
    #     else:
    #         raise TypeError(f"Unrecognized input type for encode: {type(features)}. Must be "
    #                         f"{Union[ConfigType, Sequence[ConfigType], pd.DataFrame]}.")
    #
    #     if self.feature_headers is None:
    #         self.feature_headers = xheaders
    #
    #     return encodings

    # def get_dataset(self, encodings: pd.DataFrame, labels: pd.DataFrame = None) -> xgb.DMatrix:
    #     if labels is None:
    #         return xgb.DMatrix(data=encodings, nthread=-1)
    #     else:
    #         return xgb.DMatrix(data=encodings, label=labels.sub(self.ymean, axis=1).div(self.ystd, axis=1), nthread=-1)

    def _get_simple_pipeline(self) -> sklearn.pipeline.Pipeline:
        """
        Get a Pipeline instance that can be used to train a new surrogat model. This is the simplest available pipeline
        that simply normalizes all outputs and fits an XGBoost regressor to each regressand to be predicted. HPO is
        performed by simple random search over a single set of hyperparameters common to all regressors.

        :return: pipeline
        """

        params = self.config_space.get_hyperparameters()

        # TODO: Add cache dir to speed up HPO
        onehot_columns = [p.name for p in params if isinstance(p, cs.CategoricalHyperparameter)]
        onehot_enc = sklearn.preprocessing.OneHotEncoder(drop="if_binary")
        transformers = [("OneHotEncoder", onehot_enc, onehot_columns)]
        feature_preprocessing = sklearn.compose.ColumnTransformer(transformers=transformers, remainder="passthrough")

        xgboost_estimator = xgb.sklearn.XGBRegressor(
            n_estimators=500, tree_method="gpu_hist" if self.use_gpu else "auto", **self.hyperparams)
        multi_regressor = sklearn.multioutput.MultiOutputRegressor(estimator=xgboost_estimator, n_jobs=-1)

        pipeline_steps = [
            ("preprocess", feature_preprocessing),
            ("multiout", multi_regressor)
        ]
        pipeline = sklearn.pipeline.Pipeline(steps=pipeline_steps)

        return pipeline

    # TODO: Extend input types to include ConfigType and List[ConfigType]
    def fit(self, features: pd.DataFrame, labels: pd.DataFrame, perform_hpo: bool = True, test_size: float = 0.,
            random_state: np.random.RandomState = None, hpo_iters: int = 10, num_cv_splits: int = 5):
        """ Pre-process the given dataset, fit an XGBoost model on it and return the training error. """

        # Ensure the order of labels does not get messed up
        # if self.label_headers is None:
        #     self.label_headers = labels.columns.tolist()
        # else:
        #     labels = labels[self.label_headers]

        # TODO: Check if y-scaling is needed. Add an option to enable y-scaling by building a composite estimator that
        #  works so: scale down -> real model -> scale up
        # Tree based models only benefit from normalizing the target values, not the features
        # self.ymean = labels.mean(axis=0)
        # self.ystd = labels.std(axis=0)

        # TODO: Consider replacing with GroupShuffleSplit to maintain integrity of model-wise data
        if test_size > 0.:
            xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
                features, labels, test_size=test_size, random_state=random_state, shuffle=True
            )
        elif test_size > 1.:
            raise ValueError(f"The test set fraction size 'test_size' must be in the range (0, 1), was given "
                             f"{test_size}")
        else:
            xtrain, ytrain = features, labels

        # TODO: Implement HPO with early stopping to determine correct final value for n_estimators - NASLib used a
        #  fixed value of 500, so this procedure may or may not be useful and certainly needs a reference. This is a
        #  method to prevent overfitting, analogous to cutting off NN training after a certain number of epochs.

        # TODO: Revise scoring and CV split generation
        pipeline = self._get_simple_pipeline()
        if perform_hpo:
            hpo_search_space = {f"multiout__estimator__{k}": v for k, v in self._hpo_search_space.items()}
            num_regressands = labels.columns.size
            # trainer = sklearn.model_selection.HalvingRandomSearchCV(
            #     pipeline, param_distributions=hpo_search_space, resource="multiout__estimator__n_estimators",
            #     random_state=random_state, factor=2, max_resources=self.estimators_per_output * num_regressands,
            #     min_resources=2 * num_splits * num_regressands, cv=num_splits
            # )
            trainer = sklearn.model_selection.RandomizedSearchCV(
                estimator=pipeline, param_distributions=hpo_search_space, n_iter=hpo_iters, cv=num_cv_splits,
                random_state=random_state, refit=False
            )
            search_results = trainer.fit(xtrain, ytrain)
            self.model = search_results
        else:
            self.model = pipeline.fit(xtrain, ytrain)

        if test_size > 0.:
            # TODO: Generate test accuracy
            score = self.model.score(xtest, ytest)
            return score

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """ Given some input data, generate model predictions. The input data will be properly encoded when this
        function is called. """

        ypredict = self.model.predict(features)
        return ypredict
