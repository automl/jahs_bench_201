"""
Collate all the data generated during training into one big pandas DataFrame. This script was designed specifically to
work with the directory structure defined by the class DirectoryTree. The individual DataFrame objects are expected to
contain all their data as columns with the appropriate metrics names. Theoretically, these can be anything, but the
code has been tested for the following for task and model metrics only:

tasks
-----
    - model_idx
    - model_config
    - global_seed
    - size_MB

models - [train/valid/test]
------
    - duration
    - forward_duration
    - data_load_duration
    - loss
    - acc
    - data_transfer_duration
    - backprop_duration

models - diagnostic
    - FLOPS
    - latency
    - runtime
    - cpu_percent
    - memory_ram
    - memory_swap
"""

import argparse
import logging
from pathlib import Path
import sys
from jahs_bench.tabular.lib.postprocessing import collation

_log = logging.getLogger(__name__)

if __name__ == "__main__":
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)
    collation._log.addHandler(ch)

    parser = argparse.ArgumentParser("Collates all results from the DataFrames produced after successful NASLib "
                                     "training runs into a single large DataFrame.")
    parser.add_argument("--basedir", type=Path,
                        help="The base directory for the directory tree, or the root directory within which multiple "
                             "such base directories exist, depending on whether the script is running in 'stand-alone' "
                             "or 'cluster' mode. See '--workerid' for details.")
    parser.add_argument("--file", default=None, type=Path,
                        help="The desired output filename. The filename should not carry any extension as '.pkl.gz' "
                             "will be automatically appended to it, unless the extension is already '.pkl.gz'. "
                             "Default: <basedir>/data.pkl.gz")
    # TODO: Remove the summarize and clenaup flags, this functionality has been moved to data_verification.
    parser.add_argument("--summarize", action="store_true",
                        help="When this flag is given, a summary of the current status of each run is generated and "
                             "saved on a per-task and job-level basis.")
    parser.add_argument("--cleanup", action="store_true",
                        help="When this flag is given, various files such as error descriptions and checkpoints are "
                             "verified for correctness and consistency, and deleted so as to restore the jobs to an "
                             "state.")
    # TODO: Remove anonymize, this functionality is probably useless.
    parser.add_argument("--anonymize", action="store_true",
                        help="When this flag is given, the task and model ids of the metric data are removed.")
    parser.add_argument("--workerid", type=int, default=-1,
                        help="A worker ID for launching this script on a cluster. When a non-negative value is given, "
                             "it is assumed that the script is operating in a cluster with multiple-parallel processes "
                             "and each process will choose a sub-directory within the given 'basedir' as its own base "
                             "directory based on the fidelity parameter values. Passing a negative value (default: -1) "
                             "assumes the script is running in a stand-alone fashion and will directly use the "
                             "provided 'basedir' as the directory tree's base directory.")
    parser.add_argument("--debug", action="store_true", help="When given, enables debug level output.")
    args = parser.parse_args()

    if args.debug:
        _log.setLevel(logging.DEBUG)
        collation._log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)
        collation._log.setLevel(logging.WARNING)

    if args.workerid >= 0:
        fids = {"N": [1, 3, 5], "W": [4, 8, 16], "Resolution": [0.25, 0.5, 1.0]}
        wid = args.workerid
        if wid >= 27:
            sys.exit(0)
        ids = [wid % 3, (wid // 3) % 3, (wid // 9) % 3]
        subdir = "-".join([f"{k}-{fids[k][ids[i]]}" for i, k in enumerate(["N", "W", "Resolution"])])

        basedir = args.basedir / subdir / "tasks"
        outdir = args.basedir / subdir
    else:
        basedir = args.basedir
        outdir = args.basedir
        wid = 0

    _log.info(f"Worker {wid}: Beginning metric data collation at {basedir}.")

    try:
        collated_df = collation.collate_tasks(basedir=basedir, cleanup=args.cleanup, save_summary=args.summarize,
                                              anonymize=args.anonymize)
    except Exception as e:
        _log.warning(f"Worker {wid}: Could not collate metrics DataFrames. Exiting. Cause: {str(e)}")
        sys.exit(1)

    outfile: Path = outdir / "metrics.pkl.gz" if args.file is None else args.file.resolve()
    if collated_df is None:
        _log.info(f"Worker {wid}: No valid metric data found at: {args.basedir}")
    else:
        collated_df.to_pickle(outfile)

    _log.info(f"Worker {wid}: Finished metric data collation.")
