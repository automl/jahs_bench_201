import argparse
import logging
import math
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
                             "surrogate model. The DataFrame should have a 2-level MultiIndex as its columns, with the "
                             "0-th level having the labels 'features' and 'labels' to indicate the input and output "
                             "data respectively. The DataFrame's index should be a MultiIndex with 3 levels: 'taskid', "
                             "'model_idx' and 'Epoch'.")
    parser.add_argument("--test_frac", type=float,
                        help="Fraction of the training data to be used as a test set i.e. held out and not used for "
                             "training. Must be a value in the range [0.0, 1.0). A value of 0.0 disables the creation "
                             "of a test set. ")
    parser.add_argument("--data_frac", type=float, default=1.0,
                        help="Fraction of the whole dataset to be used, a value in the range (0.0, 1.0]. Default: 1.0. "
                             "Can be set to a value lower than 1.0 to indicate that only a fraction of all the "
                             "available data is to be used. The test set (if any) is carved out of this reduced "
                             "dataset, e.g. when 'test_frac' is 0.2 and 'data_frac' is 0.8, (0.2 * 0.8 =) 16% of "
                             "all the available data will be used as a test set and (0.8 * 0.8 =) 64% of it will be "
                             "used to actually train the surrogate. When choosing a subset of data, the data is split "
                             "such that all epochs' data from the same model is counted as a single whole, e.g. if "
                             "there are 100 unique model configs' data with 100 epochs for each model, for a total of "
                             "10,000 rows, then a 'data_frac' 0.8 would choose the all 100 epochs' data of the first "
                             "80 model configs for a total of 8,000 rows.")
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
                        help="When this flag is given, enables usage of GPU accelerated model training.")
    args = parser.parse_args()
    return args


def train_surrogate(datapth: Path, test_frac: float, data_frac: float = 1.0, disable_hpo: bool = False,
                    hpo_budget: Any = None, cv_splits: int = 3, outputs: Optional[Sequence[str]] = None,
                    use_gpu: bool = False):
    """ Train a surrogate model. """

    logger.info(f"Training surrogate using data from {datapth}, using {data_frac * 100:.2f}% of the available data, "
                f"and splitting off {test_frac * 100:.2f}% of it as a test set. HPO is "
                f"{'off' if disable_hpo else 'on'}. GPU usage is {'on' if use_gpu else 'off'}."
                f"{(' Chosen output columns:' + str(outputs)) * (outputs is not None)}")
    surrogate = xgb.XGBSurrogate(use_gpu=use_gpu)
    logger.info("Fetching data.")
    data = pd.read_pickle(datapth)
    logger.info("Finished loading data.")

    if data_frac < 1.0:
        model_configs: pd.MultiIndex = data.index.droplevel("Epoch").drop_duplicates()
        model_configs_sub = model_configs[:math.floor(data_frac * model_configs.size)]
        if model_configs_sub.size == 0:
            raise ValueError(f"The given value of 'data_frac' {data_frac} is too small and caused the training dataset "
                             f"to be empty. The full dataset contains {model_configs.size} unique model configs.")
        data = data.loc[
                pd.IndexSlice[model_configs_sub.get_level_values(0), model_configs_sub.get_level_values(1), :], :]

    logger.info(f"Model training will use {data.index.size} rows of data, including the test set (if any).")
    groups = None if "groups" not in data.columns else data.groups
    features = data.features
    labels = data.labels

    if outputs is not None:
        labels = labels[outputs]

    start = time.time()
    logger.info(f"Beginning model training.")

    test_score = surrogate.fit(features, labels, groups=groups, test_size=test_frac, perform_hpo=not disable_hpo,
                               random_state=_seed, hpo_iters=hpo_budget, num_cv_splits=cv_splits)
    end = time.time()
    logger.info(f"Model training took {end - start} seconds.")
    logger.info(f"Test R2 score: {test_score}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
    cli_args = parse_cli()
    train_surrogate(**vars(cli_args))
