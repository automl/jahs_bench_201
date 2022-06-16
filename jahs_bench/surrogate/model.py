from __future__ import annotations

import copy
import logging
from functools import partial
from pathlib import Path
from typing import Union, Optional, Sequence, Tuple, Dict

import ConfigSpace
import joblib
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.compose
import sklearn.metrics
import sklearn.model_selection
import sklearn.multioutput
import sklearn.pipeline
import sklearn.preprocessing
import xgboost as xgb
import yacs.config as config
from jahs_bench.lib.core import utils as core_utils
from jahs_bench.lib.core.configspace import joint_config_space
from jahs_bench.surrogate import utils as surrogate_utils, config

_log = logging.getLogger(__name__)
ConfigType = Union[dict, ConfigSpace.Configuration]


class XGBSurrogate:
    """ A surrogate model based on XGBoost. """

    estimators_per_output: int
    hyperparams: dict
    config_space: ConfigSpace.ConfigurationSpace
    label_headers: Optional[pd.Index]
    feature_headers: Optional[pd.Series]
    trained_: bool

    __params_filename = "params.pkl.gz"
    __headers_filename = "label_headers.pkl.gz"
    __model_filename = "model.pkl.gz"
    __param_keys = ["estimators_per_output", "hyperparams", "config_space",
                    "label_headers", "feature_headers",
                    "trained_"]
    __objective = "reg:squarederror"

    _hpo_search_space = {
        "objective": [__objective],
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
            "objective": self.__objective,
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
                # "objective": "reg:squarederror",
                "objective": self.__objective,
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

    def __init__(self, config_space: Optional[
        ConfigSpace.ConfigurationSpace] = joint_config_space,
                 estimators_per_output: int = 500, use_gpu: Optional[bool] = None,
                 hyperparams: dict = None):
        """
        Initialize the internal parameters needed for the surrogate to understand the
        data it is dealing with.
        :param config_space: ConfigSpace.ConfigurationSpace
            A config space to describe what each model config looks like.
        :param estimators_per_output: int
            The number of trees that each XGB forest should boost. Experimental, will
            probably be removed in favour of a dynamic approach soon.
        :param use_gpu: bool
            A flag to ensure that a GPU is used for model training. If False (default),
            the decision is left up to XGBoost itself, which in turn depends on being
            able to detect a GPU.
        """

        self.config_space = config_space
        self.estimators_per_output = estimators_per_output
        self.hyperparams = hyperparams
        self.use_gpu = use_gpu
        self.model = None
        self.feature_headers = None
        self.label_headers = None
        self.trained_ = False

        # Both initializes some internal attributes as well as performs a sanity test
        if self.hyperparams is None:
            self.set_random_hyperparams()  # Sets default hyperparameters

    @property
    def preprocessing_pipeline(self):
        """ The pre-defined pre-processing pipeline used by the surrogate. """

        params = self.config_space.get_hyperparameters()

        # TODO: Add cache dir to speed up HPO
        # TODO: Read categorical choices from the config space
        onehot_columns = [p.name for p in params if
                          isinstance(p, ConfigSpace.CategoricalHyperparameter)]
        onehot_enc = sklearn.preprocessing.OneHotEncoder(drop="if_binary")
        transformers = [("OneHotEncoder", onehot_enc, onehot_columns)]
        prep_pipe = sklearn.compose.ColumnTransformer(transformers=transformers,
                                                      remainder="passthrough")
        return prep_pipe

    # TODO: Also handle assigning appropriate dtypes in Pandas DataFrame
    def _random_data(
        self, nconfigs: int = 10, samples_per_config: int = 100, nlabel_dims: int = 2,
        label_names: Optional[Sequence[str]] = None,
        random_state: Optional[np.random.RandomState] = None,
        config_space_constraints: Optional[Dict[str, Any]] = None) -> \
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ A debugging tool. Generate a random dataset of arbitrary size using the
        stored config space representation. Returns the randomly generated set of
        features, labels and groups as pandas DataFrame objects, in that order. A
        dictionary mapping the respective config space parameter names to appropriate
        values may be passed in 'config_space_consts' in order to restrict the sampling
        such that a subset of parameters always have the specified values. Note that this
        affects the reproducibility as well as statistical independence of the sampling
        procedure, in that these properties are only guaranteed for the modified search
        space. The resultant configurations will still be compatible with the original
        search space and any assosciated pre-processing pipelines, as long as the
        specified constant values are also compatible. """

        cs = copy.deepcopy(self.config_space)
        if config_space_constraints is not None:
            cs, _ = core_utils.adapt_search_space(original_space=cs,
                                                  opts=config_space_constraints)

        if not isinstance(random_state, np.random.RandomState):
            # Assume that rng is either None or a compatible source of entropy
            random_state = np.random.RandomState(random_state)

        cs.random = random_state

        features = cs.sample_configuration(nconfigs)
        features = np.array([list(c.get_dictionary().values()) for c in features])
        features = np.repeat(features, samples_per_config, axis=0)
        features = pd.DataFrame(features, columns=cs.get_hyperparameter_names())
        epochs = pd.Series(np.tile(np.arange(1, samples_per_config + 1), nconfigs),
                           name="epoch")
        features.loc[:, "epoch"] = epochs

        labels = random_state.random((nconfigs * samples_per_config, nlabel_dims))
        label_names = [f"Label_{i}" for i in
                       range(nlabel_dims)] if label_names is None else label_names
        labels = pd.DataFrame(labels, columns=label_names)

        groups = np.repeat(np.arange(1, nconfigs + 1), samples_per_config, axis=0)
        groups = pd.DataFrame(groups, columns=["ModelIndex"])

        return features, labels, groups

    def _get_simple_pipeline(self, multiout: bool = True) -> sklearn.pipeline.Pipeline:
        """
        Get a Pipeline instance that can be used to train a new surrogat model. This is
        the simplest available pipeline that simply normalizes all outputs and fits an
        XGBoost regressor to each regressand to be predicted. HPO is performed by simple
        random search over a single set of hyperparameters common to all regressors.

        :return: pipeline
        """

        prep_pipe = self.preprocessing_pipeline
        xgboost_estimator = xgb.sklearn.XGBRegressor(
            n_estimators=500, tree_method="gpu_hist" if self.use_gpu else "auto",
            n_jobs=1, **self.hyperparams)

        if multiout:
            multi_regressor = sklearn.multioutput.MultiOutputRegressor(
                estimator=xgboost_estimator, n_jobs=1)
            pipeline_steps = [
                ("preprocess", prep_pipe),
                ("multiout", multi_regressor)
            ]
        else:
            pipeline_steps = [
                ("preprocess", prep_pipe),
                ("estimator", xgboost_estimator)
            ]

        pipeline = sklearn.pipeline.Pipeline(steps=pipeline_steps)
        return pipeline

    def _build_pipeline(self, pipeline_config: config.CfgNode):
        """ Construct a customized pipeline using various parameters. """

        _log.debug(f"Constructing a custom pipeline using the given pipeline "
                   f"configuration.")

        # Construct the objective function
        loss_type = config.RegressionLossFuncTypes(pipeline_config.loss)
        _log.debug(f"Using the '{loss_type.name}' objective.")
        if loss_type is config.RegressionLossFuncTypes.custom:
            self.__objective = \
                surrogate_utils.custom_loss_function(pipeline_config.loss_params)
        else:
            self.__objective = loss_type.value
        self.hyperparams["objective"] = self.__objective

        # Build input preprocessing pipeline and bare-bones XGBoost estimator
        prep_pipe = self.preprocessing_pipeline
        xgboost_estimator = xgb.sklearn.XGBRegressor(
            n_estimators=500, tree_method="gpu_hist" if self.use_gpu else "auto",
            n_jobs=1, **self.hyperparams)

        # Target processing pipeline steps
        target_config = pipeline_config.target_config
        assert target_config.transform is None or isinstance(target_config.transform, str)

        if not target_config.transform:
            estimator = xgboost_estimator
        else:
            estimator = surrogate_utils.apply_target_transform(
                target_config=target_config, final_estimator=xgboost_estimator)

        # Final pipeline generation - multiout will be handled by the public API
        pipeline_steps = [
            ("preprocess", prep_pipe),
            ("estimator", estimator)
        ]

        pipeline = sklearn.pipeline.Pipeline(steps=pipeline_steps)
        _log.debug(f"Finished generating new pipeline.")
        return pipeline

    @staticmethod
    def prepare_dataset_for_training(
        features: pd.DataFrame, labels: pd.DataFrame,
        groups: Optional[pd.DataFrame] = None, test_size: float = 0.,
        random_state: Optional[np.random.RandomState] = None, num_cv_splits: int = 5,
        stratify: bool = True, strata: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
               Optional[pd.DataFrame], Optional[pd.DataFrame],
               sklearn.model_selection.BaseCrossValidator]:
        """
        Given a full dataset consisting of the input features, the labels to be
        predicted, and an optional definition of groups for the data, ascertains the way
        the data should be split into a training, validation and test set to be used for
        training a surrogate pipeline.

        :param features: pandas.DataFrame
        :param labels: pandas.DataFrame
        :param groups: pandas.DataFrame or None
            A set of additional numeric labels for each row in 'features' and 'labels'
            such that all rows with the same label are assosciated with the same model
            config and as such will be treated as one unit during all dataset shuffling
            and splitting operations. Can be omitted (i.e. set to None)
        :param test_size: float
            A real number in the range [0., 1.) to indicate what fraction of the given
            data should be split off into a test dataset. A value of 0. (default)
            indicates that no test set should be generated.
        :param random_state:
        :param num_cv_splits: int
            An integer >=2, the number of cross-validation splits to be generated from
            the training set. Can be an arbitrary number that won't affect model
            training if no Cross-Validation is actually performed during model training.
        :param stratify: bool
            When True, output strata distribution is also considered when splitting the
            data. Also consult ´strata´.
        :param strata: None or string or pandas Series
            The strata according to which splitting occurs, such that a best effort is
            made to maintain the distribution of either the groups (if given) or the
            raw data rows across the given strata. If None, and ´stratify=True´, the
            first column in the labels is used as the strata. A string can also be passed
            to indicate which column from labels DataFrame is to be used as strata.
            Otherwise, this should be a pandas Series or other iterable with a length
            corresponding to the number of rows in the dataset. No effect when
            ´stratify=False´.
        :return:
            A tuple containing the training features, test features, training labels,
            test labels, an optional set of corresponding groups for the training data
            and the cross-validation split generator, in that order. The cross-validation
            split generator can be used to split the training data into training and
            validation splits on the fly.
        """

        _log.info(
            "Generating training and test splits, and the validation split generator.")
        if test_size > 1. or test_size < 0.:
            raise ValueError(
                f"The test set fraction size 'test_size' must be in the range [0, 1), "
                f"was given {test_size}")
        elif test_size == 0.:
            _log.debug("No test split will be generated.")
            xtrain, ytrain = features, labels
            xtest, ytest = None, None
            if groups is None:
                _log.debug("No data groups were given.")
                splitter_type = sklearn.model_selection.KFold
            else:
                _log.debug("Generating training and validation splits in accordance with "
                           "the given data groups.")
                splitter_type = sklearn.model_selection.GroupKFold
        else:
            _log.debug(f"Generating a test split containing approximately "
                       f"{test_size * 100} % of the total data.")
            strata = labels.loc[:, labels.columns[0]] if strata is None else \
                labels.loc[:, strata] if isinstance(strata, str) else strata

            if groups is None:
                _log.debug("No data groups were given.")
                splitter_type = sklearn.model_selection.StratifiedShuffleSplit \
                    if stratify else sklearn.model_selection.ShuffleSplit
                test_splitter = splitter_type(n_splits=1, test_size=test_size,
                                              random_state=random_state)

                idx_train, idx_test = next(test_splitter.split(features, strata))
            else:
                _log.debug(
                    "Generating training, validation and test splits in accordance with "
                    "the given data groups.")
                splitter_type = partial(sklearn.model_selection.StratifiedGroupKFold,
                                        shuffle=True) \
                    if stratify else sklearn.model_selection.GroupShuffleSplit
                test_splitter = splitter_type(n_splits=int(1 / test_size),
                                              random_state=random_state)

                idx_train, idx_test = next(
                    test_splitter.split(features, strata, groups=groups))
                groups = groups.iloc[idx_train]

            xtrain = features.iloc[idx_train]
            xtest = features.iloc[idx_test]
            ytrain = labels.iloc[idx_train]
            ytest = labels.iloc[idx_test]
            strata = strata.iloc[idx_train] if stratify else strata

        cv = splitter_type(n_splits=num_cv_splits, random_state=random_state)

        _log.info("Dataset splits successfully generated.")
        return xtrain, xtest, ytrain, ytest, groups, strata, cv

    # TODO: Check if a train split, after test split has been taken out, would have
    #  sufficient representation in terms of categorical values (at least one occurence
    #  of each)
    def fit(self, features: pd.DataFrame, labels: pd.DataFrame,
            random_state: np.random.RandomState = None,
            pipeline_config: Optional[config.CfgNode] = None):
        """
        Pre-process the given dataset, fit an XGBoost model on it and return the training
        errors.

        :param features: DataFrame
            The inputs to the predictive model.
        :param labels: DataFrame
            The known labels the model is expected to predict for the given inputs.
        :param random_state: NumPy RandomState object or None
            A random number generator. If it's None, local system entropy is used to
            generate an RNG on the spot. The latter will break reproducibility.
        :param pipeline_config: ConfigNode or None
            A configuration object specifying how to build the training pipeline. Consult
            `XGBSurrogate._build_pipeline()` and `surrogate.constants` for more details.
        :return:
        """

        # Ensure the order of the features does not get messed up and is always accessible
        if self.feature_headers is None:
            self.feature_headers = features.columns
        else:
            features = features.loc[:, self.feature_headers]

        # Ensure the order of labels does not get messed up and is always accessible
        if self.label_headers is None:
            self.label_headers = labels.columns
        else:
            labels = labels.loc[:, self.label_headers]

        # TODO: Implement early stopping to determine correct final value for
        #  n_estimators - NASLib used a fixed value of 500, so this procedure may or may
        #  not be useful and certainly needs a reference. This is a method to prevent
        #  overfitting, analogous to cutting off NN training after a certain number of
        #  epochs.

        num_regressands = labels.columns.size
        pipeline = self._get_simple_pipeline(multiout=num_regressands > 1) \
            if not pipeline_config else self._build_pipeline(pipeline_config)

        self.model = pipeline.fit(features, labels)
        self.trained_ = True

        ypred_train = self.predict(features)
        train_r2 = sklearn.metrics.r2_score(labels, ypred_train)
        train_mse = sklearn.metrics.mean_squared_error(labels, ypred_train)
        scores = {
            "train_r2": train_r2,
            "train_mse": train_mse,
        }

        return scores

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """ Given some input data, generate model predictions. The input data will be
        properly encoded when this function is called. """

        features = features.loc[:, self.feature_headers]
        ypredict = self.model.predict(features)
        ypredict = pd.DataFrame(ypredict, columns=self.label_headers)
        return ypredict

    def dump(self, outdir: Path, protocol: int = 0):
        """ Save a trained surrogate to disk so that it can be loaded up later. """

        params = {k: self.__getattribute__(k) for k in self.__param_keys}
        joblib.dump(params, outdir / self.__params_filename, protocol=protocol)
        if self.trained_:
            self.label_headers.to_series().to_pickle(outdir / self.__headers_filename,
                                                     protocol=protocol)
            joblib.dump(self.model, outdir / self.__model_filename, protocol=protocol)

    @classmethod
    def load(cls, outdir: Path) -> XGBSurrogate:
        """ Load a previously saved surrogate from disk and return it. """

        params: dict = joblib.load(outdir / cls.__params_filename)
        surrogate = cls()
        for k, v in params.items():
            surrogate.__setattr__(k, v)

        if surrogate.trained_:
            label_headers: pd.Series = pd.read_pickle(outdir / cls.__headers_filename)
            model = joblib.load(outdir / cls.__model_filename)

            surrogate.label_headers = pd.Index(label_headers)
            surrogate.model = model

        return surrogate
