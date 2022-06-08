import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
# Make sure that the system path contains the correct repository
from jahs_bench.surrogate import model, utils

_seed = 3501623856

_log = logging.getLogger(__name__)


def generate_splits(datadir: Path, datafile: str, test_size: float,
                    valid_size: Optional[float] = None,
                    outputs: Optional[Sequence[str]] = None, seed: Optional[int] = _seed):
    _log.info(f"Setting up the output directory and random state using the seed {seed}.")
    subdir = datadir / f"valid-{valid_size}-test-{test_size}"
    subdir.mkdir(exist_ok=False, parents=False)
    random_state = np.random.RandomState(seed)

    _log.info("Loading dataset from disk.")
    datapth = datadir / datafile
    features, labels, groups, strata = utils.adjust_dataset(
        datapth=datapth, outputs=outputs, fillna=True)

    if valid_size > 0.:
        train_set_frac = 1 - min(test_size, 1 - test_size)
        valid_frac = valid_size / train_set_frac
        _log.info(f"Using {valid_frac * 100} % of the training set to achieve an "
                  f"effective validation set fraction of {valid_size}.")
    else:
        valid_frac = 0.2  # Placeholder, this won't actually generate a valid split

    _log.info(f"Generating training and test splits using test set fraction {test_size}.")
    if test_size > 0.5:
        xtest, xtrain, ytest, ytrain, groups, strata, cv = \
            model.XGBSurrogate.prepare_dataset_for_training(
                features=features, labels=labels, groups=groups, test_size=1 - test_size,
                random_state=random_state, num_cv_splits=int(1 / valid_frac),
                stratify=True, strata=strata
            )
    else:
        xtrain, xtest, ytrain, ytest, groups, strata, cv = \
            model.XGBSurrogate.prepare_dataset_for_training(
                features=features, labels=labels, groups=groups, test_size=test_size,
                random_state=random_state, num_cv_splits=int(1 / valid_frac),
                stratify=True, strata=strata
            )

    _log.info(f"The test split contains {xtest.shape[0]} data points.")

    if valid_size > 0.:
        _log.info(f"Generating validation split.")
        idx_train, idx_valid = next(cv.split(xtrain, strata, groups=groups))
        xvalid = xtrain.iloc[idx_valid]
        yvalid = ytrain.iloc[idx_valid]
        xtrain = xtrain.iloc[idx_train]
        ytrain = ytrain.iloc[idx_train]
        groups = groups.iloc[idx_train]
        strata = strata.iloc[idx_train]
        _log.info(f"The training and validation splits contain {xtrain.shape[0]} and "
                  f"{xvalid.shape[0]} data points respectively.")
    else:
        _log.info(f"The training split contains {xtrain.shape[0]} data points.")

    _log.info("Saving training, validation and test sets to disk")
    pd.concat({
        "features": xtrain, "labels": ytrain,
        "groups": groups, "strata": strata
    }, axis=1).to_pickle(subdir / "train_set.pkl.gz")

    pd.concat({"features": xvalid, "labels": yvalid}, axis=1).to_pickle(
        subdir / "valid_set.pkl.gz")
    pd.concat({"features": xtest, "labels": ytest}, axis=1).to_pickle(
        subdir / "test_set.pkl.gz")

    _log.info(f"Finished generating splits for test size {test_size}.")


def parse_cli():
    parser = argparse.ArgumentParser(
        "Script to prepare the performance dataset for use in training a surrogate model."
    )
    parser.add_argument("--datadir", type=Path,
                        help="The directory where all the relevant data is stored as "
                             "well as where the data splits will be stored.")
    parser.add_argument("--datafile", type=str,
                        help="The name of the file inside `datadir` which contains the "
                             "full dataset that should be split up.")
    parser.add_argument("--test_size", type=float,
                        help="The fraction of the full dataset that should be used for "
                             "generating the test split.")
    parser.add_argument("--valid_size", type=float,
                        help="The fraction of the full dataset that should be used for "
                             "generating the validation split.")
    parser.add_argument("--outputs", type=str, default=None,
                        nargs=argparse.REMAINDER,
                        help="Strings, separated by spaces, that indicate which of the "
                             "available metrics in the dataset should be used for "
                             "generating the splits. If not given, all metrics present "
                             "in the full dataset are used.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    _log.setLevel(logging.INFO)
    generate_splits(**vars(args))

