"""
This script is intended to verify that a given stage of the benchmark is reproducible.
The underlying idea is that for each sampled configuration, it should be possible to
reconstruct the network using the saved model configuration, load the second to last
checkpoint, resume training for a small number of epochs and cross-check the generated
metrics against the saved metric values from the previous run (i.e. the latest metrics
DataFrame). A subset of all recorded metrics are deterministic and the rest are
stochastic. This script should report the observed difference in stochastic metric values
as well as catch instances where the deterministic metric values differed.
"""

import argparse
import logging
from pathlib import Path

from jahs_bench.tabular.lib.core import constants

duration_metrics = [m for m in constants.standard_model_dataset_metrics
                    if "duration" in metric]
data_splits = ["train", "valid", "test"]
stochastic_metrics = [(s, m) for s in data_splits for m in duration_metrics] + \
                     [("train", m) for m in constants.extra_model_training_metrics] + \
                     [("diagnostic", m) for m in ["latency", "runtime"]]

deterministic_metrics = [(s, m) for s in data_splits for m in ["loss", "acc"]] + \
                        [("diagnostic", m) for m in ["FLOPS", "size_MB"]]


_log = logging.getLogger(__name__)


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rootdir", type=Path, default=Path().cwd(),
                        help="Path to the root directory where all the tasks' output was stored. Task-specific "
                             "sub-directories will be created here if needed. Note that this is NOT the same as the "
                             "'basedir' of a DirectoryTree object as multiple DirectoryTree objects will be created "
                             "as and when needed.")
    parser.add_argument("--tmp_rootdir", type=Path, default="/tmp/nashpo_benchmarks/rootdir",
                        help="A directory which will be used to store temporary data, "
                             "essentially creating a temporary partial copy of the metrics and checkpoints of the original ")
    parser.add_argument("--workerid", type=int,
                        help="An offset from 0 for this worker within the current job's allocation of workers. This "
                             "does NOT correspond to a single 'taskid' but rather the portfolio of configurations that "
                             "this worker will evaluate.")
    parser.add_argument("--workerid-offset", type=int, default=0,
                        help="An additional fixed offset from 0 for this worker that, combined with the value of "
                             "'--worker', gives the overall worker ID in the context of a job that has been split into "
                             "multiple smaller parts.")
    parser.add_argument("--nworkers", type=int,
                        help="The total number of workers that are expected to concurrently perform data verification "
                             "in this root directory. This is used to coordinate the workers such that there is no "
                             "overlap in their portfolios.")
    parser.add_argument("--datadir", type=Path,
                        help="The directory where all datasets are expected to be stored.")
    parser.add_argument("--configs", type=Path,
                        help="Path to a pickled Pandas DataFrame (*.pkl.gz) that contains the full configuration of "
                             "each known model that should be verified. The index of each such configuration should be "
                             "the unique tuple (taskid, model_idx), where the corresponding values will be used to "
                             "construct corresponding directory trees.")
    parser.add_argument("--nepochs", type=int, default=None,
                        help="Only used in the 'verify' mode of operation. The number of epochs of data that should be "
                             "generated and verified using each loaded checkpoint. This implies that, for example, in "
                             "a model with 5 pairs of checkpoints and metric logs that have already been verified to "
                             "be in sync, the first 4 checkpoints will each be loaded up, the model trained for "
                             "'nepochs' epochs, and the corresponding entries in the last 4 metrics logs will be "
                             "cross-checked.")
    parser.add_argument("--budget", type=int,
                        help="Time budget for each worker in seconds. This is used to approximate whether or not a "
                             "worker has enough time to perform one more operation without any risk to any data.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")

    return parser
