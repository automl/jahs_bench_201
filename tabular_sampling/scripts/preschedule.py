"""
This script is intended to be used for pre-computing a more efficient schedule for distributing model configuration
evaluations across the workers in a job allocation using data that has been generated from a smaller run over all the
planned samples in the search space.
"""


import argparse
import itertools
import logging
import pandas as pd
from pathlib import Path
from string import Template
import sys
from typing import Union

import tabular_sampling.lib.postprocessing.metric_df_ops
from tabular_sampling.clusterlib import prescheduler as sched_utils
from tabular_sampling.lib.core.utils import AttrDict
from tabular_sampling.lib.postprocessing.metric_df_ops import get_configs
from tabular_sampling.distributed_nas_sampling import run_task, get_tranining_config_from_args, _seed
from tabular_sampling.lib.core import constants

_log = logging.getLogger(__name__)

def _handle_debug(args):
    if args.debug:
        _log.setLevel(logging.DEBUG)
        sched_utils._log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)
        sched_utils._log.setLevel(logging.INFO)


def _isolate_training_args(args: argparse.Namespace) -> str:
    """ Given parsed CLI arguments, isolates the values of arguments corresponding to the training config and
    re-constructs the corresponding CLI arguments. """
    training_config = vars(args)
    training_args = []
    actions = {
        "store_true": True,
        "store_false": False
    }
    for k, v in constants.training_config.items():
        if k not in training_config or training_config[k] is None:
            # No value given. If this is optional, leave it empty.
            if "action" not in v and "default" not in v:
                # This is not a flag but it also does not have a default value
                raise RuntimeError(f"Cannot parse the given training config parameters into a valid CLI arguments "
                                   f"string, missing value for required parameter {k}")
        else:
            if "action" in v and "store" in v["action"]:
                # This is a flag
                if actions[v["action"]] == training_config[k]:
                    training_args.append(f"--{k}")
                else:
                    continue
            else:
                # This is a key-value pair
                training_args.append(f"--{k}={training_config[k]}")
    return training_args


def default_training_config() -> AttrDict:
    """ Generate a training config using the default arguments. """

    default_parser = argparse.ArgumentParser()
    for k, v in constants.training_config.items():
        default_parser.add_argument(f"--{k}", **v)

    default_args = default_parser.parse_args([])
    return get_tranining_config_from_args(default_args)


def estimate_requirements(args):
    # TODO: Complete and standardize this procedure. For now, use manually generated estimates.
    raise NotImplementedError("This functionality will be properly built and re-structured at a later date.")


def init_directory_tree(args):
    _handle_debug(args)
    _log.debug("Starting sub-program: initialize directory tree.")

    taskid_levelname = constants.MetricDFIndexLevels.taskid.value
    profile_path: Path = args.profile

    try:
        profile: pd.DataFrame = pd.read_pickle(profile_path)
        basedirs: pd.Series = profile.job_config["basedir"]
    except Exception as e:
        raise RuntimeError(f"Failed to read the directory structure from {profile_path}.") from e

    if basedirs is None:
        _log.error(f"No data found in {basedirs_file}.")
        sys.exit(-1)

    sched_utils.prepare_directory_structure(basedirs=basedirs, rootdir=args.rootdir)

    _log.info("Pre-sampling model configs.")

    training_config = default_training_config()
    nsamples = basedirs.index.to_frame(index=False).groupby(taskid_levelname).max()
    if isinstance(nsamples, pd.DataFrame):
        nsamples = nsamples[nsamples.columns[0]].rename("nsamples")

    fidelities = profile.model_config
    dataset = constants.Datasets.__members__.get(args.dataset)

    ntasks = nsamples.index.size
    for i, taskid in enumerate(nsamples.index, start=1):
        fidelity = fidelities.xs(taskid, level=taskid_levelname).iloc[0]
        basedir = args.rootdir / basedirs.xs(taskid, level=taskid_levelname).iloc[0]
        run_task(basedir=basedir, taskid=taskid,
                 train_config=training_config, dataset=dataset, datadir=args.datadir, local_seed=_seed,
                 global_seed=None, debug=args.debug, generate_sampling_profile=True, nsamples=nsamples[taskid],
                 portfolio_pth=None, cycle_portfolio=False,
                 opts=fidelity.to_dict())

        if i % (5 * max(ntasks // 100, 1)) == 0:
            # Report progress every time approximately 5% of the tasks have been fully pre-sampled.
            _log.info(f"Pre-sampling progress:\t{100 * i / ntasks:=6.2f}%\t({i}/{ntasks} tasks)")

    _log.debug("Finished sub-program: initialize directory tree.")


def generate_jobs(args):

    _handle_debug(args)

    profile_path: Path = args.profile
    try:
        profile: pd.DataFrame = pd.read_pickle(profile_path)
        per_epoch_runtimes: pd.Series = profile.required.runtime
        cpus_per_worker: pd.Series = profile.required.cpus
    except Exception as e:
        raise RuntimeError(f"Failed to read the sampling profile from {profile_path}.") from e

    if cpus_per_worker.nunique() != 1:
        _log.warning(f"The given profile is heterogenous in the number of CPUs required per worker. The current state "
                     f"of this script does not support such profiles. It is advisable to break up the profile into "
                     f"multiple parts that are partwise homogenous in the number of CPUs required per worker. ")
    else:
        cpus_per_worker: int = cpus_per_worker.max()
        assert isinstance(cpus_per_worker, int), f"The number of CPUs per worker must be an integer value, was " \
                                                 f"{cpus_per_worker} of type {type(cpus_per_worker)}."

    if ("status", "nepochs") in profile.columns:
        remaining_epochs: pd.Series = profile.status.nepochs - args.epochs
    else:
        remaining_epochs: int = args.epochs

    estimated_runtimes: pd.Series = per_epoch_runtimes * remaining_epochs
    profile.loc[:, ("required", "runtime")] = estimated_runtimes

    job_config = sched_utils.JobConfig(
        cpus_per_worker=cpus_per_worker, cpus_per_node=args.cpus_per_node, nodes_per_job=args.nodes_per_job,
        timelimit=args.timelimit * 60
    )
    workers = sched_utils.allocate_work(
        job_config=job_config, profile=profile, cpuh_utilization_cutoff=args.cpuh_utilization_cutoff,
        cap_job_timelimit=args.dynamic_timelimit
    )

    sched_utils.save_worker_portfolios(workers=workers, portfolio_dir=args.portfolio_dir)

    with open(args.template_dir / "job.template") as fp:
        job_template = Template(fp.read())

    with open(args.template_dir / "config.template") as fp:
        config_template = Template(fp.read())

    ctr = itertools.count(start=1)
    workerid_offset = 0

    for _ in workers[::job_config.workers_per_job]:
        jobid = next(ctr)
        job_name = f"job-{jobid}"
        rootdir = str(args.rootdir)
        job_str = job_template.substitute(
            rootdir=rootdir, script_dir=args.script_dir, job_name=job_name, **job_config.template_kwargs
        )
        srun_str = config_template.substitute(
            rootdir=rootdir, workerid_offset=workerid_offset, portfolio_dir=str(args.portfolio_dir),
            datadir=str(args.datadir), training_config_args=" ".join(_isolate_training_args(args))
        )

        with open(args.script_dir / f"{job_name}.job", "w") as fp:
            fp.write(job_str)

        with open(args.script_dir / f"{job_name}.config", "w") as fp:
            fp.write(srun_str)

        workerid_offset += job_config.workers_per_job



def argument_parser():

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--rootdir", type=Path,
                               help="This is the directory where the outputs of the jobs will be stored i.e. WORK. "
                                    "The base directory for each DirectoryTree will be "
                                    "'<rootdir>/<fidelity_dir>/tasks', where 'fidelity_dir' is generated by joining "
                                    "the names and values of the relevant fidelity parameters in a string separated by "
                                    "'-', e.g. 'N-1-W-8-Resolution-1.0'.")
    parent_parser.add_argument("--profile", type=Path,
                               help="Path to a *.pkl.gz file which will be either read from or written to, referring "
                                    "to the relevant sampling profile. When used with the command 'estimate', this "
                                    "file will be written to. When used with either 'init' or 'gen', it will be read "
                                    "from.")
    # TODO: This can be further optimized - it is only needed in the case when a model is actually trained
    parent_parser.add_argument("--datadir", type=Path,
                               help="The directory where all datasets are expected to be stored.")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Sub-commands", required=True,
        description="Pre-scheduling consists of three broad steps: 1. estimate- estimate the resource requirements and "
                    "sampling procedure, 2. initialize: initialize the directory tree for the estimated sampling "
                    "procedure, 3. generate: new bundled jobs to handle the actual evaluation. Steps 1 and 3 may be "
                    "repeated as more data becomes available."
    )

    ## Step 1 - Estimate Sampling Profile

    subparser_est = subparsers.add_parser(
        "estimate", aliases=["est"], parents=[parent_parser],
        help="Use the available metric data and desired computation parameters to estimate how many resources will be "
             "required for the desired evaluations as well as how the sampling procedure should be distributed to "
             "optimize resource utilization."
    )
    subparser_est.set_defaults(func=estimate_requirements)

    ## Step 2 - Initialize Directory Tree

    subparser_init = subparsers.add_parser(
        "initialize", aliases=["init"], parents=[parent_parser],
        help="Use the predicted total evaluation profile to generate the directory structure that will be needed for "
             "carrying out the expected calculations. Performing this step in advance allows the main jobs to run with "
             "the assumption that each individual model's parent directories already exist, fixed model configurations, "
             "training configurations and random seeds have been generated, and prevents a number of issues with "
             "multi-processing."
    )
    subparser_init.set_defaults(func=init_directory_tree)
    subparser_init.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    subparser_init.add_argument("--dataset",
                                choices=list(constants.Datasets.__members__.keys()),
                                help=f"The name of which dataset is to be used for model training and evaluation. Only "
                                     f"one of the provided choices can be used.")

    ## Step 3 - Generate bundled jobs

    subparser_gen = subparsers.add_parser(
        "generate", aliases=["gen"], parents=[parent_parser],
        help="Use the existing resource requirement estimates and the given job resource availability parameters to "
             "generate concrete jobs that can be submitted to a SLURM cluster. This re-packages the estimated workload "
             "into job bundles of a desired size, e.g. 'n' nodes per job."
    )
    subparser_gen.set_defaults(func=generate_jobs)

    subparser_gen.add_argument("--cpus_per_node", type=int,
                               help="Job configuration - the number of CPUs that are to be expected in each node of "
                                    "the cluster.")
    subparser_gen.add_argument("--nodes_per_job", type=int,
                               help="The number of nodes that are available per independent job.")
    subparser_gen.add_argument("--timelimit", type=int,
                               help="The maximum amount of time (in minutes) that a single job is allowed to run for.")
    subparser_gen.add_argument("--dynamic_timelimit", action="store_true",
                               help="When this flag is given, the time limit of the job is dynamically adjusted to "
                                    "maximize the CPUh utilization. The dynamically adjusted time limit can only be "
                                    "lower than that specified by '--timelimit'. If this flag is omitted, the job will "
                                    "have exactly the value of '--timelimit' as its time limit.")
    subparser_gen.add_argument("--cpuh_utilization_cutoff", type=float, default=0.8,
                               help="Recommended minimum fraction of total allocated CPUh that should be actively used "
                                    "for computation. Generates a warning when the job allocation's expected CPUh "
                                    "utilization is below this value. Default: 0.8")
    subparser_gen.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    subparser_gen.add_argument("--template_dir", type=Path,
                               help="Path to a directory which contains templates using which the job files and their "
                                    "relevant srun config files are generated. The templates for jobs and configs "
                                    "should be named 'job.template' and 'config.template' respectively.")
    subparser_gen.add_argument("--script_dir", type=Path,
                               help="This is the directory where the generated job scripts and their respective srun "
                                    "configuration files will be stored.")
    subparser_gen.add_argument("--portfolio_dir", type=Path,
                               help="Path to a directory from where each worker will be able to store its own "
                                    "allocated portfolio of configurations to evaluate.")

    # Unpack the training config into CLI flags.
    for k, v in constants.training_config.items():
        subparser_gen.add_argument(f"--{k}", **v)

    return parser

if __name__ == "__main__":

    # Setup this module's logger
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)
    _log.setLevel(logging.INFO)

    sched_utils._log.addHandler(ch)
    sched_utils._log.setLevel(logging.INFO)

    ## Parse CLI
    args = argument_parser().parse_args()
    args.func(args)
