import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence
import yaml

import pandas as pd
import scipy.stats
import sklearn.metrics
from jahs_bench.surrogate import model

_default_test_set_fn = "test_set.pkl.gz"
_default_test_pred_fn = "test_pred.pkl.gz"
_log = logging.getLogger(__name__)


def load_test_set(testset_file: Path, outputs: Optional[Sequence[str]] = None) -> \
                  pd.DataFrame:
    assert testset_file.exists()
    _log.info(f"Attempting to read test set from file {testset_file}.")
    test_set = pd.read_pickle(testset_file)
    if outputs is not None:
        _log.info(f"Restricting test set to the given outputs: {outputs}.")
        outputs = outputs if isinstance(outputs, list) else [outputs] \
            if isinstance(outputs, str) else list(outputs)  # Coerce into a list

        assert test_set["labels"].columns.intersection(outputs).size == len(outputs), \
            f"The given set of outputs is not a subset of the outputs present in the " \
            f"test set: {test_set['labels'].columns.tolist()}."

        sel = test_set[["features"]].columns.tolist() + \
              test_set.loc[:, ("labels", outputs)].columns.tolist()
        test_set = test_set.loc[:, sel]

    _log.info(f"Successfully loaded test set of shape {test_set.shape}.")
    return test_set


def evaluate_test_set(test_set: pd.DataFrame, model_dir: Path,
                      outputs: Optional[Sequence[str]] = None,
                      save_dir: Optional[Path] = None) -> pd.DataFrame:
    """ Evaluate the model on the given test set and returns the predictions as a pandas
    DataFrame. """

    _log.info("Extracting the input features and expected labels from the test set.")
    xtest = test_set.loc[:, "features"]
    ytest = test_set.loc[:, "labels"]
    assert xtest.shape[0] == ytest.shape[0]

    _log.info(f"Test set has {xtest.shape[0]} data points.")

    if outputs is None:
        outputs = ytest.columns.tolist()

    ypred = {}
    for output in outputs:
        _log.info(f"Preparing to load surrogate for output: {output}")
        model_subdir = model_dir / str(output)
        surrogate = model.XGBSurrogate.load(model_subdir)

        _log.info(f"Generating surrogate predictions.")
        ypred[output] = surrogate.predict(xtest)

    _log.info("Concatenating predictions.")
    ypred = pd.concat(ypred, axis=1)
    if save_dir is not None:
        # TODO: If 'exist_ok=False' is to be used, it should be done at the very start
        save_dir.mkdir(exist_ok=True, parents=True)
        ypred.to_pickle(save_dir / _default_test_pred_fn)

    return ypred


def score_predictions(test_set: pd.DataFrame, ypred: pd.DataFrame):
    """ Given a test set and predictions made on that test set, returns the Kendall-Tau
    rank correlation (KT) and Coefficient of Determination (R2) scores. """

    _log.info("Extracting the input features and expected labels from the test set.")
    xtest = test_set.loc[:, "features"]
    ytest = test_set.loc[:, "labels"]
    assert xtest.shape[0] == ytest.shape[0]

    _log.info(f"Generating quality of fit scores for {ypred.shape[0]} samples and "
              f"{ypred.shape[1]} outputs.")
    scores = {}
    for output in ytest.columns:
        r2_score = sklearn.metrics.r2_score(ytest[output], ypred[output])

        # No need to adjust for minimization when comparing two series of the same metric.
        ranks_test = ytest[output].rank()
        ranks_pred = ypred[output].rank()
        # kt_corr, kt_p = scipy.stats.kendalltau(ytest[output], ypred[output])
        kt_corr, kt_p = scipy.stats.kendalltau(ranks_test, ranks_pred)
        scores[output] = {
            "R2": float(r2_score),
            "KT": [float(kt_corr), float(kt_p)]
        }

    _log.info(f"Generated scores:\n{scores}")
    return scores


def main(testset_file: Path, model_dir: Optional[Path] = None,
         outputs: Optional[Sequence[str]] = None, scores_only: bool = False,
         predictions_only: bool = False, save_dir: Optional[Path] = None):
    assert testset_file is not None
    test_set = load_test_set(testset_file, outputs)

    if scores_only:
        assert save_dir is not None
        ypred = pd.read_pickle(save_dir / "test_pred.pkl.gz")
    else:
        if predictions_only:
            assert save_dir is not None, \
                "When 'predictions_only' is True, 'save_dir' cannot be None."
        ypred = evaluate_test_set(test_set, model_dir, outputs, save_dir)

    if not predictions_only:
        scores = score_predictions(test_set, ypred)
        if save_dir is not None:
            with open(save_dir / "scores.yaml", "w") as fp:
                yaml.safe_dump(scores, fp)

    _log.info(f"Finished.")


def parse_cli():
    parser = argparse.ArgumentParser(
        "Script to generate scores on a test set for a trained surrogate model."
    )
    parser.add_argument("--testset-file", type=Path, default=None,
                        help="A path to the file which contains the test set.")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="The directory where the trained surrogate models for each "
                             "relevant output are stored. Optional only when "
                             "--scores-only is given.")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="An optional directory where the pandas DataFrame of "
                             "predictions made by this script will be stored in a file "
                             "called 'test_pred.pkl.gz' and the scores will be stored in "
                             "a file called 'scores.yaml'. If not provided, the scores "
                             "and predictions are not saved to disk. Must be given if "
                             "--scores-only is given.")
    parser.add_argument("--scores-only", action="store_true", default=False,
                        help="When this flag is given, predictions are not generated "
                             "from a surrogate model. Instead, predictions from a "
                             "previous run are expected to be readable from "
                             "'save-dir/test_pred.pkl.gz'. Scores are generated using "
                             "these predictions. When this flag is given, there is no "
                             "need to provide --model_dir, but --save-dir must be given. "
                             "If this flag is not given, any previously generated "
                             "predictions will get overwritten by newly generated "
                             "predictions.")
    parser.add_argument("--predictions-only", action="store_true", default=False,
                        help="When this flag is given, only predictions are generated "
                             "from a surrogate model, without calculating the "
                             "assosciated scores. When this flag is given, both "
                             "--model_dir and --save-dir must be given. "
                             "If this flag is given, any previously generated "
                             "predictions will get overwritten by newly generated "
                             "predictions.")
    parser.add_argument("--outputs", type=str, default=None,
                        nargs=argparse.REMAINDER,
                        help="Strings, separated by spaces, that indicate which of the "
                             "available metrics in the dataset should be used for "
                             "generating the predictions. If not given, all metrics "
                             "present in the full dataset are used. Note that a model "
                             "for each output is expected to be present in 'save_dir'.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    _log.setLevel(logging.INFO)
    main(**vars(args))
