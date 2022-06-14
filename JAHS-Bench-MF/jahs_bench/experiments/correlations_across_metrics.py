""" Given a dataset of performance metrics, calculate the Kendall-Tau rank correlations
for each metric w.r.t. every other metric.

Process:
1. Accept as input a dataset that has been post-processed into wide-format, with
top-level columns "features" and "labels".
2. Filter out only the metrics specified, if needed.
3. For each pair of ranked metrics, calculate KT scores.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Sequence
import logging
_log = logging.getLogger(__name__)
from tabular_sampling.lib.postprocessing import surrogate_analysis as analysis


def main(metrics_file: Path, save_dir: Path, outputs: Optional[Sequence[str]]):
    _log.info(f"Reading wide-format metrics file from {metrics_file}.")
    wide_metrics = pd.read_pickle(metrics_file)

    if outputs is not None:
        _log.info(f"Restricting correlation calculation to the given outputs: {outputs}.")
        outputs = outputs if isinstance(outputs, list) else [outputs] \
            if isinstance(outputs, str) else list(outputs)  # Coerce into a list

        check = wide_metrics["labels"].columns.intersection(outputs)
        assert check.size  == len(outputs), \
            f"The given set of outputs is not a subset of the outputs present in the " \
            f"test set: {test_set['labels'].columns.tolist()}."

        sel = test_set[["features"]].columns.tolist() + \
              test_set.loc[:, ("labels", outputs)].columns.tolist()
        test_set = test_set.loc[:, sel]

    fidelity = analysis.Fidelity(N=[1, 3, 5], W=[4, 8, 16], Resolution=[0.25, 0.5, 1.0],
                                 Epoch=list(range(1, 201)))

    _log.info(f"Preparing data.")
    filtered = analysis.get_filtered_data(wide_metrics, fidelity)

    _log.info(f"Generating ranks and correlations.")
    correlations = analysis.get_correlations(filtered)

    save_pth = save_dir / "correlations.pkl.gz"
    _log.info(f"Saving correlations dataframe to {save_pth}.")
    correlations.to_pickle(save_pth)

def parse_cli():
    parser = argparse.ArgumentParser(
        "Script to generate scores on a test set for a trained surrogate model."
    )
    parser.add_argument("--metrics-file", type=Path, default=None,
                        help="A path to the file which contains the wide-format metrics "
                             "data using which correlations needs to be calculated.")
    parser.add_argument("--save-dir", type=Path,
                        help="A directory where the pandas DataFrame of correlations and "
                             "p-values made by this script will be stored in a file "
                             "called 'correlations.pkl.gz'.")
    parser.add_argument("--outputs", type=str, default=None,
                        nargs=argparse.REMAINDER,
                        help="Strings, separated by spaces, that indicate which of the "
                             "available metrics in the dataset should be used for "
                             "generating the correlations. If not given, all metrics "
                             "present in the full dataset are used.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                        datefmt="%m/%d %H:%M:%S")
