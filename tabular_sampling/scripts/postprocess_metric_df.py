import argparse
import logging
from pathlib import Path
import sys

from tabular_sampling.lib.postprocessing import metric_df_ops as metric_ops, _log as pproc_log

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
                        help="The base directory, same as what was used by the training script.")
    parser.add_argument("--outdir", default=None, type=Path,
                        help="The desired output directory where post-processes data will be stored as various "
                             "pickled DataFrame objects. Default: <outdir>/postproc")
    parser.add_argument("--debug", action="store_true", help="When given, enables debug level output.")
    parser.add_argument("--workerid", type=int, default=-1, help="A worker ID for launching this script on a cluster.")
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
        pproc_log.setLevel(logging.INFO)

    # fids = {"N": [1, 3, 5], "W": [4, 8, 16], "Resolution": [0.25, 0.5, 1.0]}
    # wid = args.workerid
    # if wid >= 27:
    #     sys.exit(0)
    # ids = [wid % 3, (wid // 3) % 3, (wid // 9) % 3]
    # subdir = "-".join([f"{k}-{fids[k][ids[i]]}" for i, k in enumerate(["N", "W", "Resolution"])])
    #
    # basedir = args.basedir / subdir
    basedir = args.basedir

    df = metric_ops.load_metric_df(basedir=basedir)

    fidelity_confs = [("model_config", f) for f in metric_ops.fidelity_params]
    nsamples = metric_ops.get_nsamples(basedir=None, df=df, groupby=fidelity_confs, index=metric_ops.fidelity_params)
    runtimes = metric_ops.get_runtimes(basedir=None, df=df, reduce_epochs=True, extra_durations=None)
    acc_200epochs = metric_ops.get_accuracies(basedir=None, df=df, display=False, filter_epochs=200)
    acc_all_epochs = metric_ops.get_accuracies(basedir=None, df=df, display=False, filter_epochs=-1)

    outdir = basedir / "postproc" if args.outdir is None else args.outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=False)

    nsamples.to_pickle(outdir / "nsamples.pkl.gz")
    runtimes.to_pickle(outdir / "runtimes.pkl.gz")
    acc_200epochs.to_pickle(outdir / "acc_200epochs.pkl.gz")
    acc_all_epochs.to_pickle(outdir / "acc_all_epochs.pkl.gz")
