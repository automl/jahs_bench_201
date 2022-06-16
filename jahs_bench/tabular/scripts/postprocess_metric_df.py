import argparse
import logging
from pathlib import Path
import sys

from jahs_bench.tabular.lib.core.constants import fidelity_params
from jahs_bench.tabular.lib.postprocessing import _log as pproc_log
from jahs_bench.tabular.lib.postprocessing import metric_df_ops as metric_ops

_log = logging.getLogger(__name__)

if __name__ == "__main__":
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)
    pproc_log.addHandler(ch)

    parser = argparse.ArgumentParser("Collates all results from the DataFrames produced after successful NASLib "
                                     "training runs into a single large DataFrame.")
    parser.add_argument("--basedir", type=Path,
                        help="The base directory for the directory tree, or the root directory within which multiple "
                             "such base directories exist, depending on whether the script is running in 'stand-alone' "
                             "or 'cluster' mode. See '--workerid' for details.")
    parser.add_argument("--outdir", default=None, type=Path,
                        help="The desired output directory where post-processes data will be stored as various "
                             "pickled DataFrame objects. Default: <outdir>/postproc")
    parser.add_argument("--debug", action="store_true", help="When given, enables debug level output.")
    parser.add_argument("--workerid", type=int, default=-1,
                        help="A worker ID for launching this script on a cluster. When a non-negative value is given, "
                             "it is assumed that the script is operating in a cluster with multiple-parallel processes "
                             "and each process will choose a sub-directory within the given 'basedir' as its own base "
                             "directory based on the fidelity parameter values. Passing a negative value (default: -1) "
                             "assumes the script is running in a stand-alone fashion and will directly use the "
                             "provided 'basedir' as the directory tree's base directory.")
    parser.add_argument("--log", type=Path, default=None,
                        help="A log file to which output logs will be saved. Specifying this turns off logging to "
                             "stdout and stderr (some initial logs may still be sent to stdout and stderr).")
    args = parser.parse_args()

    if args.log is not None:
        fh = logging.FileHandler(args.log.resolve())
        fh.setFormatter(fmt)
        _log.removeHandler(ch)
        _log.addHandler(fh)
        pproc_log.removeHandler(ch)
        pproc_log.addHandler(fh)

    if args.debug:
        _log.setLevel(logging.DEBUG)
        pproc_log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)
        pproc_log.setLevel(logging.WARNING)

    if args.workerid >= 0:
        fids = {"N": [1, 3, 5], "W": [4, 8, 16], "Resolution": [0.25, 0.5, 1.0]}
        wid = args.workerid
        if wid >= 27:
            sys.exit(0)
        ids = [wid % 3, (wid // 3) % 3, (wid // 9) % 3]
        subdir = "-".join([f"{k}-{fids[k][ids[i]]}" for i, k in enumerate(["N", "W", "Resolution"])])

        basedir = args.basedir / subdir
    else:
        basedir = args.basedir

    _log.info(f"Worker {wid}: Beginning metric data postprocessing at {basedir}.")
    try:
        df = metric_ops.load_metric_df(basedir=basedir)
    except Exception as e:
        _log.warning(f"Worker {wid}: Could not load metric DataFrame. Exiting. Cause: {str(e)}")
        sys.exit(1)

    fidelity_confs = [("model_config", f) for f in fidelity_params]
    configs = metric_ops.get_configs(basedir=None, df=df)

    nepochs_200 = metric_ops.get_nepochs(basedir=None, df=df, filter_epochs=200)
    nepochs_all = metric_ops.get_nepochs(basedir=None, df=df, filter_epochs=-1)

    acc_all_epochs = metric_ops.get_accuracies(basedir=None, df=df, include_validation=True)
    # acc_all_epochs = acc_all_epochs.join([configs])
    acc_200epochs = acc_all_epochs.loc[nepochs_200.index]

    loss_all_epochs = metric_ops.get_losses(basedir=None, df=df, include_validation=True)
    # loss_all_epochs = loss_all_epochs.join([configs])
    loss_200epochs = loss_all_epochs.loc[nepochs_200.index]

    nsamples = metric_ops.get_nsamples(basedir=None, df=df, groupby=fidelity_confs, index=fidelity_params)
    runtimes = metric_ops.get_runtimes(basedir=None, df=df, reduce_epochs=True, extra_durations=None)
    # runtimes = runtimes.join([configs])
    remaining_runtimes = metric_ops.estimate_remaining_runtime(basedir=None, df=df, max_epochs=200)
    # remaining_runtimes = remaining_runtimes.join([configs])

    outdir = basedir / "postproc" if args.outdir is None else args.outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=False)

    configs.to_pickle(outdir / "configs.pkl.gz")
    nsamples.to_pickle(outdir / "nsamples.pkl.gz")
    runtimes.to_pickle(outdir / "runtimes.pkl.gz")
    nepochs_all.to_pickle(outdir / "nepochs.pkl.gz")
    remaining_runtimes.to_pickle(outdir / "remaining_runtimes.pkl.gz")
    acc_200epochs.to_pickle(outdir / "acc_200epochs.pkl.gz")
    acc_all_epochs.to_pickle(outdir / "acc_all_epochs.pkl.gz")
    loss_200epochs.to_pickle(outdir / "loss_200epochs.pkl.gz")
    loss_all_epochs.to_pickle(outdir / "loss_all_epochs.pkl.gz")

    _log.info(f"Worker {wid}: Finished postprocessing.")
