"""
This script is intended to be used for pre-computing a more efficient schedule for distributing model configuration
evaluations across the workers in a job allocation using data that has been generated from a smaller run over all the
planned samples in the search space.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from string import Template
import shutil
import sys
import time

from jahs_bench.tabular.clusterlib import prescheduler as sched_utils
from jahs_bench.tabular.lib.core.utils import AttrDict
from jahs_bench.tabular.distributed_nas_sampling import run_task, get_tranining_config_from_args, _seed
from jahs_bench.tabular.lib.core import constants

from jahs_bench.tabular.lib.naslib.utils.logging import setup_logger


_log = setup_logger(name='tabular_sampling')


def generate_worker_portfolio(workerid: int, nworkers: int, full_profile: pd.DataFrame) -> pd.DataFrame:
    _log.debug(f"Found {full_profile.index.size} configurations to construct portfolio from.")

    if isinstance(full_profile, pd.DataFrame):
        portfolio = full_profile.loc[full_profile.index[workerid::nworkers], :]
    elif isinstance(full_profile, pd.Series):
        portfolio = full_profile[full_profile.index[workerid::nworkers]]
    else:
        raise RuntimeError("Unexpected control flow sequence. This piece of code should be unreachable. Given "
                           f"profile of type {type(full_profile)}.")
    _log.debug(f"Created portfolio of {portfolio.index.size} configs.")
    return portfolio


# noinspection PyShadowingNames,PyProtectedMember
def _handle_debug(args):
    if args.debug:
        _log.setLevel(logging.DEBUG)
    else:
        _log.setLevel(logging.INFO)


# noinspection PyShadowingNames
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


# noinspection PyUnusedLocal,PyShadowingNames
def estimate_requirements(args):
    # TODO: Complete and standardize this procedure. For now, use manually generated estimates.
    raise NotImplementedError("This functionality will be properly built and re-structured at a later date.")


# noinspection PyShadowingNames
def init_directory_tree(args):
    _handle_debug(args)
    _log.debug("Starting sub-program: initialize directory tree.")

    ## Estimated time budget requirement
    start = time.time()
    budget = args.budget - 60

    taskid_levelname = constants.MetricDFIndexLevels.taskid.value
    profile_path: Path = args.profile
    workerid = args.workerid_offset + args.workerid

    try:
        profile = pd.read_pickle(profile_path)
        assert profile.index.is_unique, "Each row index in the profile must be unique throughout the profile."
        basedirs: pd.Series = profile.job_config["basedir"]
    except Exception as e:
        raise RuntimeError(f"Failed to read the directory structure from {profile_path}.") from e

    if basedirs is None:
        _log.error(f"Profile {profile_path} contains no data on the basedirs.")
        sys.exit(-1)

    if args.create_tree:
        sched_utils.prepare_directory_structure(
            basedirs=generate_worker_portfolio(workerid=workerid, nworkers=args.nworkers, full_profile=basedirs),
            rootdir=args.rootdir
        )
        return

    _log.info("Pre-sampling model configs.")

    training_config = default_training_config()
    nsamples = basedirs.index.to_frame(index=False).groupby(taskid_levelname).max()
    if isinstance(nsamples, pd.DataFrame):
        nsamples: pd.Series = nsamples[nsamples.columns[0]].rename("nsamples")

    # Only consider this worker's allocation.
    nsamples = generate_worker_portfolio(workerid=workerid, nworkers=args.nworkers, full_profile=nsamples)

    fidelities = profile.model_config
    dataset = constants.Datasets.__members__.get(args.dataset)

    opts = {}
    if args.opts is not None:
        if isinstance(args.opts, list):
            i = iter(args.opts)
            opts = {k: v for k, v in zip(i, i)}
        else:
            raise RuntimeError(f"Unable to parse the search space overrides {args.opts}")

    ## Handle the checkpointing of this worker's portfolio
    worker_chkpt_subdir = "worker_chkpts"
    chkpt_name = f"preschedule_init_worker.pkl.gz"
    assert args.rootdir.exists(), f"The specified root directory {args.rootdir} does not exist."
    worker_chkpt_dir: Path = args.rootdir / worker_chkpt_subdir / str(workerid)
    worker_chkpt_dir.mkdir(exist_ok=True, parents=True)
    chkpt_filepath = worker_chkpt_dir / chkpt_name

    done = None
    if chkpt_filepath.exists():
        try:
            done = pd.read_pickle(chkpt_filepath)
            nsamples = nsamples[done == False]
            fidelities = fidelities.loc[nsamples.index, :]
        except Exception as e:
            _log.info(f"Ran into an error while trying to read {chkpt_filepath}: {str(e)}")
            done = None

    if done is None:
        done = pd.Series(False, index=nsamples.index)
        done.to_pickle(chkpt_filepath)

    end = time.time()
    duration = end - start
    _log.info(f"Worker {workerid}: Setting up the directory tree consumed {duration} seconds.")
    budget -= duration
    start = end
    max_time = 0.

    ntasks = nsamples.index.size
    for i, taskid in enumerate(nsamples.index, start=1):
        if budget < 1.2 * max_time:
            break

        fidelity = fidelities.xs(taskid, level=taskid_levelname).iloc[0]
        basedir = args.rootdir / basedirs.xs(taskid, level=taskid_levelname).iloc[0]
        run_task(basedir=basedir, taskid=taskid,
                 train_config=training_config, dataset=dataset, datadir=args.datadir, local_seed=_seed,
                 global_seed=None, debug=args.debug, generate_sampling_profile=True, nsamples=nsamples[taskid],
                 portfolio_pth=None, cycle_portfolio=False, logger=_log,
                 opts={**fidelity.to_dict(), **opts})

        if i % (5 * max(ntasks // 100, 1)) == 0:
            # Report progress every time approximately 5% of the tasks have been fully pre-sampled and checkpoint
            # progress.
            _log.info(f"Worker {workerid}: Pre-sampling progress:\t{100 * i / ntasks:=6.2f}%\t({i}/{ntasks} tasks)")
            _log.info(f"Worker {workerid}: Maximum time per sampling operation: {max_time} seconds. Remaining Budget: "
                      f"{budget} seconds.")
            done.iloc[:i] = True
            shutil.copyfile(chkpt_filepath, f"{chkpt_filepath}.bak")
            done.to_pickle(chkpt_filepath)

        ## Update duration estimate
        end = time.time()
        duration = end - start
        max_time = max(max_time, duration)
        budget -= duration
        start = end

    _log.info(f"Worker {workerid}: Finished sub-program: initialize directory tree.")


# noinspection PyShadowingNames
def generate_jobs(args):
    _handle_debug(args)

    profile_path: Path = args.profile
    try:
        profile: pd.DataFrame = pd.read_pickle(profile_path)
        assert profile.index.is_unique, "Each row index in the profile must be unique throughout the profile."
        cpus_per_worker: pd.Series = profile.required.cpus
    except Exception as e:
        raise RuntimeError(f"Failed to read the sampling profile from {profile_path}.") from e

    if cpus_per_worker.nunique() != 1:
        _log.warning(f"The given profile is heterogeneous in the number of CPUs required per worker. The current state "
                     f"of this script does not support such profiles. It is advisable to break up the profile into "
                     f"multiple parts that are part-wise homogenous in the number of CPUs required per worker. ")

    cpus_per_worker: int = cpus_per_worker.max()
    assert isinstance(cpus_per_worker, int), f"The number of CPUs per worker must be an integer value, was " \
                                             f"{cpus_per_worker} of type {type(cpus_per_worker)}."

    job_config = sched_utils.JobConfig(
        cpus_per_worker=cpus_per_worker, cpus_per_node=args.cpus_per_node, nodes_per_job=args.nodes_per_job,
        timelimit=args.timelimit * 60
    )
    jobs = sched_utils.allocate_work(
        job_config=job_config, profile=profile, epochs=args.epochs,
        cpuh_utilization_cutoff=args.cpuh_utilization_cutoff, dynamic_timelimit=args.dynamic_timelimit,
        dynamic_nnodes=args.dynamic_nnodes, remainder_pth=args.profile_remainder, worker_id_offset=args.workerid_start,
        squish_configs=args.squish_configs
    )

    with open(args.template_dir / "job.template") as fp:
        job_template = Template(fp.read())

    with open(args.template_dir / "config.template") as fp:
        config_template = Template(fp.read())

    for jobid, job in enumerate(jobs, start=args.jobid_start):
        _log.info(f"Saving files for job #{jobid}.")
        job_name = f"job-{jobid}"
        rootdir = str(args.rootdir)

        _log.info(f"Saving portfolios.")
        portoflio_fn = job.save_worker_portfolios(jobid=jobid, portfolio_dir=args.portfolio_dir)

        job_str = job_template.substitute(
            rootdir=rootdir, script_dir=args.script_dir, job_name=job_name, **job.config.template_kwargs
        )
        srun_str = config_template.substitute(
            rootdir=rootdir, workerid_offset=job.worker_id_offset, portfolio=str(portoflio_fn),
            datadir=str(args.datadir), training_config_args=" ".join(_isolate_training_args(args))
        )

        _log.info("Saving job and config.")
        with open(args.script_dir / f"{job_name}.job", "w") as fp:
            fp.write(job_str)

        with open(args.script_dir / f"{job_name}.config", "w") as fp:
            fp.write(srun_str)

    _log.info(f"Finished generating all job files. To continue adding jobs to this job allocation, use the Job ID "
              f"Offset {args.jobid_start + len(jobs)}")


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
             "the assumption that each individual model's parent directories already exist, fixed model "
             "configurations, training configurations and random seeds have been generated, and prevents a number of "
             "issues with multi-processing."
    )
    subparser_init.set_defaults(func=init_directory_tree)
    subparser_init.add_argument("--debug", action="store_true", help="Enable debug mode (very verbose) logging.")
    subparser_init.add_argument("--dataset",
                                choices=list(constants.Datasets.__members__.keys()),
                                help=f"The name of which dataset is to be used for model training and evaluation. Only "
                                     f"one of the provided choices can be used.")
    subparser_init.add_argument("--workerid", type=int,
                                help="An offset from 0 for this worker within the current job's allocation of workers. "
                                     "This does NOT correspond to a single 'taskid' but rather the portfolio of "
                                     "configurations that this worker will handle.")
    subparser_init.add_argument("--workerid-offset", type=int, default=0,
                                help="An additional fixed offset from 0 for this worker that, combined with the value "
                                     "of '--worker', gives the overall worker ID in the context of a job that has been "
                                     "split into multiple smaller parts.")
    subparser_init.add_argument("--nworkers", type=int,
                                help="The total number of workers that are expected to concurrently perform data "
                                     "verification in this root directory. This is used to coordinate the workers such "
                                     "that there is no overlap in their portfolios.")
    subparser_init.add_argument("--budget", type=int,
                                help="Time budget for each worker in seconds. This is used to approximate whether or "
                                     "not a worker has enough time to perform one more operation without any risk to "
                                     "any data. The script reserves 60 seconds of the budget for initialization, i.e. "
                                     "it always assumes a budget of 60 seconds less than what was specified here.")
    subparser_init.add_argument("--create_tree", action="store_true",
                                help="When given, also creates the required directory trees under rootdir. Otherwise, "
                                     "assumes that the directory tree already exists. This should be done exactly once "
                                     "with a single worker.")
    subparser_init.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                                help="A variable number of optional keyword arguments provided as 2-tuples, each "
                                     "potentially corresponding to a hyper-parameter in the search space. If a match "
                                     "is found, that hyper-parameter is excluded from the search space and fixed to "
                                     "the given value instead. This also overrides the respective values read in from "
                                     "profiles.")

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
    subparser_gen.add_argument("--dynamic_nnodes", action="store_true",
                               help="When this flag is given, the number of nodes requested by each job is dynamically "
                                    "adjusted to maximize the CPUh utilization. The dynamically adjusted number of "
                                    "nodes can only be lower than that specified by '--nodes_per_job'. If this flag is "
                                    "omitted, each job will request exactly '--nodes_per_job' nodes, even if there "
                                    "aren't enough busy workers to use them.")
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
    subparser_gen.add_argument("--profile_remainder", type=Path, default=None,
                               help="Full or relative (to the current working directory) path to *.pkl.gz file, in "
                                    "which any configurations from the profile that could not be fit into the given "
                                    "resource constraints will be stored. If the file exists, it is overwritten. If "
                                    "all configurations specified in the sampling profile have bee allocated to a "
                                    "worker's portfolio, this file is not created, but any existing file will remain"
                                    "untouched.")
    subparser_gen.add_argument("--jobid_start", type=int, default=0,
                                help="A fixed offset from 0 for all jobs produced by this run. This allows multiple "
                                     "executions of this script to be used to chain together multiple jobs without "
                                     "them interfering with each other by, e.g., continuously changing the input "
                                     "profile.")
    subparser_gen.add_argument("--workerid_start", type=int, default=0,
                                help="A fixed offset from 0 for the workers of all jobs produced by this run. This "
                                     "allows multiple executions of this script to be used to chain together multiple "
                                     "jobs without them interfering with each other by, e.g., continuously changing "
                                     "the input profile.")
    subparser_gen.add_argument("--squish_configs", action="store_true",
                               help="When this flag is given, any configs that do not fit into the allocated time "
                                    "budget will be squished into the job anyways so they can run for as long as "
                                    "possible.")

    # Unpack the training config into CLI flags.
    for k, v in constants.training_config.items():
        subparser_gen.add_argument(f"--{k}", **v)

    return parser


if __name__ == "__main__":
    ## Parse CLI
    args = argument_parser().parse_args()
    args.func(args)
