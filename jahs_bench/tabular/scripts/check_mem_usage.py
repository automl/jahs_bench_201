"""
Test whether or not the maximum expected memory load on a particular dataset is viable for a given resource allocation.
This script launches the most expensive known architecture (all Ops are set to ConvBN3x3) for the fidelity assosciated
with each launched worker id for a specified dataset and attempts to train it for 5 epochs. This can also be used to
generate initial theoretical upper bounds on per-epoch training time for each fidelity bucket.
"""

import argparse
import logging
from pathlib import Path
import sys
from jahs_bench.tabular.lib.core import datasets as dataset_lib
from jahs_bench.tabular.lib.core.constants import Datasets
from jahs_bench.tabular.lib.core.utils import AttrDict
from jahs_bench.tabular import distributed_nas_sampling
from jahs_bench.tabular.search_space.constants import OP_NAMES

_log = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test whether or not the maximum expected memory load on a particular dataset is "
                                     "viable for a given resource allocation.")
    parser.add_argument("--rootdir", type=Path,
                        help="The root directory within which multiple base directories will exist. See '--workerid' "
                             "for details.")
    parser.add_argument("--workerid", type=int, default=0,
                        help="A worker ID for launching this script on a cluster. When a non-negative value is given, "
                             "it is assumed that the script is operating in a cluster with multiple parallel processes "
                             "and each process will choose a sub-directory within the given 'rootdir' as its own base "
                             "directory based on the fidelity parameter values. Raises an error when a negative value "
                             "is given.")
    parser.add_argument("--dataset", type=str, choices=list(Datasets.__members__.keys()),
                        help="The name of which dataset is to be used for model training and evaluation. Only one of "
                             "the provided choices can be used.")
    parser.add_argument("--datadir", type=Path, default=dataset_lib.get_default_datadir(),
                        help="The directory where all datasets are expected to be stored.")
    parser.add_argument("--debug", action="store_true", help="When given, enables debug level output.")
    args = parser.parse_args()

    if args.debug:
        _log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)

    wid = args.workerid

    if wid < 0:
        raise ValueError(f"Worker id must be a non-negative integer, was {args.workerid}.")

    fmt = logging.Formatter(f"*Worker {wid}* [%(asctime)s] %(name)s %(levelname)s: "
                            f"%(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)

    fids = {"N": [1, 3, 5], "W": [4, 8, 16], "Resolution": [0.25, 0.5, 1.0]}
    if wid >= 27:
        sys.exit(0)
    n, w, r = fids["N"][wid % 3], fids["W"][(wid // 3) % 3], fids["Resolution"][(wid // 9) % 3]
    ids = [n, w, r]
    subdir = "-".join([f"{k}-{ids[i]}" for i, k in enumerate(fids.keys())])

    basedir: Path = args.rootdir / subdir / "tasks"
    assert args.rootdir.exists() and args.rootdir.is_dir()
    basedir.mkdir(parents=True, exist_ok=True)

    dataset = Datasets.__members__[args.dataset]
    train_config = AttrDict(epochs=5, batch_size=256, use_grad_clipping=False, split=True, warmup_epochs=0,
                            disable_checkpointing=False, checkpoint_interval_seconds=None, checkpoint_interval_epochs=1)

    op_name = "ConvBN3x3"
    op_index = OP_NAMES.index(op_name)
    base_opts = " ".join([" ".join([k, str(ids[i])]) for i, k in enumerate(fids.keys())] + ["Optimizer SGD"])
    arch_opts = " ".join([f"Op{i} {op_index}" for i in range(1, 7)])  # Set all edge operations
    opts = " ".join([base_opts, arch_opts]).split(" ")

    _log.info(f"Worker {wid}: Beginning memory usage check at {basedir} for dataset {dataset.value[0]}.")

    try:
        distributed_nas_sampling.run_task(
            basedir=basedir, taskid=args.workerid, train_config=train_config, dataset=dataset, datadir=args.datadir,
            local_seed=distributed_nas_sampling._seed, global_seed=None, debug=False, generate_sampling_profile=False,
            nsamples=1, portfolio_pth=None, cycle_portfolio=False, logger=_log, opts=opts
        )
    except Exception as e:
        _log.warning(f"Worker {wid}: Ran into an error while running the model at {basedir}. Exiting. Cause: {str(e)}")
        sys.exit(1)

    _log.info(f"Worker {wid}: Finished memory usage test.")
