import argparse
import logging
import time
from pathlib import Path
from typing import Any, Sequence, Optional

import pandas as pd

from tabular_sampling.surrogate import xgb

logger = logging.getLogger(__name__)
_seed = 3501623856


def parse_cli():
    parser = argparse.ArgumentParser("Train a surrogate model based on the given data.")
    parser.add_argument("--datapth", type=Path,
                        help="Either the full path or path relative to the current working directory to a *.pkl.gz "
                             "file containing the full pickled DataFrame that is to be used for training/testing the "
                             "surrogate model. The DataFrame should have been converted to a compatible long-format "
                             "DataFrame and not be in the raw metrics DataFrame format.")
    parser.add_argument("--test_frac", type=float,
                        help="Fraction of the training data to be used as a test set i.e. held out and not used for "
                             "training. Must be a value in the range [0.0, 1.0). A value of 0.0 disables the creation "
                             "of a test set. ")
    parser.add_argument("--save_dir", type=Path, default=None,
                        help="Path (absolute or relative to the current working directory) to a directory where the "
                             "trained model will be saved. Care should be taken that a single leaf directory in any "
                             "directory tree should be used for saving only one model. Re-using the same directory for "
                             "saving multiple models will result in the model files overwriting each other. If this is "
                             "not specified, the model will not be saved.")
    parser.add_argument("--disable_hpo", action="store_true",
                        help="When this flag is given, no Hyper-parameter Optimization is performed on the surrogate's "
                             "hyper-parameters. Instead, only the default set of hyper-parameters is used.")
    # TODO: Revise HPO Algorithm
    parser.add_argument("--hpo_budget", type=int, default=10,
                        help="The budget of the HPO procedure. The interpretation of this parameter depends on the "
                             "type of HPO algorithm used.")
    parser.add_argument("--cv_splits", type=int, default=3,
                        help="The number of cross-validation splits to be used in case HPO is enabled.")
    parser.add_argument("--outputs", nargs="*", default=None,
                        help="A subset of the column labels under 'labels' in the DataFrame of the dataset which "
                             "specifies that the surrogate should be trained to predict only these outputs. If not "
                             "specified, all available outputs are used (default).")
    parser.add_argument("--use_gpu", action="store_true",
                        help="When this flag is given, enables usage of GPU accelerated model training. Otherwise, the "
                             "choice of using GPU acceleration is left up to the model.")
    parser.add_argument("--fillna", action="store_true",
                        help="When this flag is given, a best effort is made to automatically fill any NaN values in "
                             "the labels with approrpriate numbers - 0 for any accuracy metrics and a very large "
                             "number for duration/loss metrics.")
    args = parser.parse_args()
    return args


def train_surrogate(datapth: Path, test_frac: float, disable_hpo: bool = False,
                    hpo_budget: Any = None, cv_splits: int = 3, outputs: Optional[Sequence[str]] = None,
                    use_gpu: bool = False, save_dir: Optional[Path] = None, fillna: bool = False):
    """ Train a surrogate model. """

    logger.info(f"Training surrogate using data from {datapth} and splitting off {test_frac * 100:.2f}% of it as a "
                f"test set. HPO is {'off' if disable_hpo else 'on'}. GPU usage is {'on' if use_gpu else 'off'}."
                f"{(' Chosen output columns: ' + str(outputs)) * (outputs is not None)}")
    surrogate = xgb.XGBSurrogate(use_gpu=use_gpu)
    logger.info("Fetching data.")
    data = pd.read_pickle(datapth)
    logger.info("Finished loading data.")

    logger.info(f"Model training will use {data.index.size} rows of data, including the test set (if any).")
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
            logger.info("Found no NaN values in the labels.")
        else:
            logger.info(f"Found {subdf.shape[0]} rows of data with NaN values in them. Attempting to fill NaN values.")
            fillvals = {}
            for c in sel.index:
                if sel[c]:
                    if "acc" in c:
                        fillvals[c] = 0.
                    elif "loss" in c or "duration" in c:
                        fillvals[c] = 100 * labels[c].max()
                    else:
                        raise RuntimeError(f"Could not find appropriate fill value to replace NaNs with for metric: "
                                           f"{c}")
            logger.info(f"Filling NaN values according to the mapping: {fillvals}")
            labels.fillna(fillvals, inplace=True)

    start = time.time()
    logger.info(f"Beginning model training.")

    test_scores = surrogate.fit(features, labels, groups=groups, test_size=test_frac, perform_hpo=not disable_hpo,
                                random_state=_seed, hpo_iters=hpo_budget, num_cv_splits=cv_splits, stratify=True,
                                strata=strata)
    end = time.time()
    logger.info(f"Model training took {end - start} seconds.")
    logger.info(f"Final scores: {test_scores}")
    if save_dir is not None:
        surrogate.dump(save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    cli_args = parse_cli()
    train_surrogate(**vars(cli_args))
