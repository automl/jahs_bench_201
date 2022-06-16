import argparse
import functools
import logging
from pathlib import Path
from typing import Tuple, Optional

import joblib
import neps
import pandas as pd
import scipy.stats
from jahs_bench.surrogate import model, config as cfg

_log = logging.getLogger(__name__)

xgb_hp_space = {
    "max_depth": neps.IntegerParameter(
        lower=1, upper=15, default=6, default_confidence="low"
    ),
    "min_child_weight": neps.IntegerParameter(
        lower=1, upper=10, default=1, default_confidence="low"
    ),
    "colsample_bytree": neps.FloatParameter(
        lower=0., upper=1., log=False, default=1., default_confidence="low"
    ),
    "learning_rate": neps.FloatParameter(
        lower=0.001, upper=0.5, log=True, default=0.3, default_confidence="low"
    ),
    "colsample_bylevel":  neps.FloatParameter(
        lower=0., upper=1., log=False, default=1., default_confidence="low"
    ),
    "sigmoid_k": neps.FloatParameter(
        lower=0.01, upper=2., log=True, default=1., default_confidence="low"
    )
}

def train_surrogate(working_directory: Path, train_data: pd.DataFrame,
                    valid_data: pd.DataFrame, **config_dict):
    """

    :param working_directory:
    :param train_data: pandas DataFrame
        A pandas DataFrame object with 2-level MultiIndex columns, containing the
        training data. Level 0 should contain the columns "features" and "labels". Level
        1 can contain arbitrary columns.
    :param valid_data: pandas DataFrame
        A pandas DataFrame object with 2-level MultiIndex columns, containing the
        validation data. Level 0 should contain the columns "features" and "labels".
        Level 1 can contain arbitrary columns, but they should match those in
        `train_data`.
    :param config_dict: keyword-arguments
        These specify the various hyperparameters to be used for the model.
    :return:
    """

    xtrain = train_data["features"]
    ytrain = train_data["labels"]

    xvalid = valid_data["features"]
    yvalid = valid_data["labels"]

    _log.info(f"Preparing to train surrogate.")
    pipeline_config = cfg.default_pipeline_config.clone()
    xgb_params = config_dict.copy()
    sigmoid_k = xgb_params.pop("sigmoid_k", None)

    if sigmoid_k is not None:
        pipeline_config.defrost()
        pipeline_config.target_config.params.params[1].k = float(sigmoid_k)
        pipeline_config.freeze()

    surrogate = model.XGBSurrogate(hyperparams=xgb_params, use_gpu=True)

    _log.info("Training surrogate.")
    random_state = None
    scores = surrogate.fit(xtrain, ytrain, random_state=random_state,
                           pipeline_config=pipeline_config)
    _log.info(f"Trained surrogate has scores: {scores}")

    modeldir = working_directory / "xgb_model"
    _log.info(f"Saving trained surrogate to disk at {str(modeldir)}")
    modeldir.mkdir()
    surrogate.dump(modeldir)
    config_file = modeldir / "pipeline_config.pkl.gz"
    joblib.dump(pipeline_config, config_file, protocol=4)

    _log.info(f"Generating validation scores.")
    ypred = surrogate.predict(xvalid)
    valid_score = scipy.stats.kendalltau(yvalid, ypred)

    _log.info(f"Trained surrogate has validation score: {valid_score}")
    # Return negative KT correlation since NEPS minimizes the loss
    return -valid_score.correlation

def load_data(datadir: Path, output: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _log.info("Loading training data.")
    train_data: pd.DataFrame = pd.read_pickle(datadir / "train_set.pkl.gz")
    _log.info(f"Loaded training data of shape {train_data.shape}")

    _log.info("Loading validation data.")
    valid_data: pd.DataFrame = pd.read_pickle(datadir / "valid_set.pkl.gz")
    _log.info(f"Loaded validation data of shape {valid_data.shape}")

    features = train_data["features"].columns
    labels = train_data["labels"].columns

    assert features.difference(valid_data["features"].columns).size == 0, \
        "Mismatch between the input features in the training and validation sets."
    assert labels.difference(valid_data["labels"].columns).size == 0, \
        "Mismatch between the output labels in the training and validation sets."

    if output is not None:
        assert output in labels, f"The chosen prediction label {output} is not present " \
                                 f"in the training data."
        selected_cols = train_data[["features"]].columns.tolist() + [("labels", output)]
        train_data = train_data.loc[:, selected_cols]
        valid_data = valid_data.loc[:, selected_cols]

    return train_data, valid_data

def perform_hpo(working_directory: Path, datadir: Path, output: str,
                max_evaluations_total: int = 5,):
    _log.info(f"Performing HPO using the working directory {working_directory}, using "
              f"the data at {datadir}, over {max_evaluations_total} evaluations.")

    train_data, valid_data = load_data(datadir=datadir, output=output)
    pipeline = functools.partial(train_surrogate, train_data=train_data,
                                 valid_data=valid_data)
    neps.run(
        run_pipeline=pipeline,
        pipeline_space=xgb_hp_space,
        working_directory=working_directory,
        max_evaluations_total=max_evaluations_total,
    )

    _log.info("Finished.")

def parse_cli():
    parser = argparse.ArgumentParser(
        "Use the NEPS framework to train a surrogate model by iteratively optimizing "
        "the hyperparameters."
    )
    parser.add_argument("--working_directory", type=Path,
                        help="A working directory to be used by NEPS for various tasks "
                             "that require writing to disk.")
    parser.add_argument("--datadir", type=Path,
                        help="Path to the directory where the cleaned, tidy training "
                             "and validation data splits to be used are stored.")
    parser.add_argument("--output", type=str,
                        help="Which of the available performance metrics to train the "
                             "surrogate for.")
    parser.add_argument("--max_evaluations_total", type=int, default=5,
                        help="Number of evaluations that this NEPS worker should "
                             "perform.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    _log.setLevel(logging.INFO)
    args = parse_cli()
    perform_hpo(**vars(args))
