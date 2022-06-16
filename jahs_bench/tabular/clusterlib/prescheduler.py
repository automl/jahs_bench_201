"""
This library contains various functions that are intended to help generate a pre-computed schedule for distributing
a large number of model evaluations across a set number of jobs with certain properties. The intention is to generate
a unique portfolio file for each unique and independent worker in a distributed computing environment.
"""

import itertools
import logging
import math
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, Iterator, Tuple, Union, Dict

from jahs_bench.tabular.lib.core import constants, utils

_log = logging.getLogger(__name__)
fidelity_params = constants.fidelity_params
fidelity_types = constants.fidelity_types
logdir_name = "logs"


class JobConfig(object):
    """ A container for various properties of the cluster that will be used to help compute the work schedule. """

    cpus_per_worker: int
    cpus_per_node: int
    nodes_per_job: int
    timelimit: float

    def __init__(self, cpus_per_worker=1, cpus_per_node=48, nodes_per_job=3072, timelimit=2 * 24 * 3600):
        self.cpus_per_worker = cpus_per_worker
        self.cpus_per_node = cpus_per_node
        self.nodes_per_job = nodes_per_job
        self.timelimit = timelimit

    @property
    def workers_per_node(self) -> int:
        return self.cpus_per_node // self.cpus_per_worker

    @property
    def workers_per_job(self) -> int:
        return self.nodes_per_job * self.workers_per_node

    @property
    def template_kwargs(self) -> dict:
        """ A dict of keyword arguments that should provide most of the details needed to fill in a job template. """
        return {
            "cpus_per_task": self.cpus_per_worker,
            # "cpus_per_node": self.cpus_per_node,
            "nodes_per_job": self.nodes_per_job,
            "tasks_per_node": self.workers_per_node,
            "timelimit": self.timestr(self.timelimit),
        }

    @staticmethod
    def timestr(secs: float) -> str:
        """ Generates a human-readable and SLURM-compatible string representing the given duration in seconds as
        DD-HH:MM, where DD is the number of days, HH is the number of hours and MM is the number of minutes.
        Seconds-level precision is omitted since it is unlikely to be very useful. """

        minutes = math.ceil(secs / 60)
        hours = minutes // 60
        minutes %= 60
        days = hours // 24
        hours %= 24
        return f"{days}-{hours:02}:{minutes:02}"


class WorkerConfig(object):
    """ Container of properties for each worker's work allocation. """

    worker_id: int
    portfolio_dir: Optional[Path]
    portfolio: Optional[pd.Series]

    def __init__(self, worker_id: int, portfolio_dir: Optional[Path] = None, portfolio: Optional[pd.DataFrame] = None):
        self.worker_id = worker_id
        self.portfolio_dir = portfolio_dir
        self.portfolio = portfolio

    @property
    def portolio_file(self) -> Path:
        assert self.portfolio_dir is not None, "No portfolio directory specified, cannot generate portfolio file name."
        return self.portfolio_dir / f"portfolio_{self.worker_id}.pkl.gz"

    def load_portfolio(self):
        if self.portfolio is not None:
            _log.warning(f"Existing portfolio of worker {self.worker_id} will be overwritten by portfolio read from "
                         f"{self.portolio_file}")
        self.portfolio: pd.Series = pd.read_pickle(self.portolio_file)

    def iterate_portfolio(self, start: int = 0, stop: int = None, step: int = 1, rootdir: Optional[Path] = None) -> \
            Iterator[Tuple[int, int, Path]]:
        """ Returns an iterator over this worker's portfolio, consisting of a list of 3-tuples, each consisting of
        an integer task ID, an integer model index, and a path to the base directory where all the metrics and
        checkpoints of this configuration's previous evaluations are stored. The optional Path-like 'rootdir' can be
        specified to read all the saved base directory paths to be treated as relative to the specified 'rootdir'. """

        index = self.portfolio.index[start:stop:step]
        pths = self.portfolio[index]
        if rootdir is None:
            pths = [Path().cwd() / p for p in pths]
        else:
            pths = [(rootdir / p).resolve() for p in pths]

        return iter([(*c, p) for c, p in zip(index, pths)])

    def __hash__(self):
        return hash(self.worker_id)


WorkAssignment = Dict[WorkerConfig, Tuple[float, Sequence[Tuple[int, int]]]]


class JobAllocation:
    def __init__(self, config: JobConfig, worker_id_offset: int = 0, assignment: Optional[WorkAssignment] = None,
                 dynamic_nnodes: bool = True, dynamic_timelimit: bool = True):
        """ A container for a single job's work assignment. Used primarily for bundling together workers operating on
        the same resource assignment (JobConfig) that have been assigned some portfolio into a single job file. """
        self.config = JobConfig(cpus_per_worker=config.cpus_per_worker, cpus_per_node=config.cpus_per_node,
                                nodes_per_job=config.nodes_per_job, timelimit=config.timelimit)
        self.worker_id_offset = worker_id_offset
        self.assignment = assignment
        self.dynamic_nnodes = dynamic_nnodes
        self.dynamic_timelimit = dynamic_timelimit

    @property
    def assignment(self) -> Dict[WorkerConfig, Tuple[float, Sequence[Tuple[int, int]]]]:
        """ """
        return self._assignment

    @assignment.setter
    def assignment(self, value: WorkAssignment):
        if value is None:
            # Initialize with an empty work assignment
            self.workers: Sequence[WorkerConfig] = [WorkerConfig(worker_id=self.worker_id_offset + i)
                                                    for i in range(self.config.workers_per_job)]
            self._assignment = {w: [0., []] for w in self.workers}
        else:
            assert isinstance(value, dict), f"The work assignment of a JobAllocation must be a dictionary of the " \
                                            f"form {WorkAssignment}."
            self.workers: Sequence[WorkerConfig] = list(value.keys())
            self._assignment = value

    @property
    def cpuh_demand(self) -> float:
        """ Total requested CPUh """
        return self.config.workers_per_job * self.config.cpus_per_worker * self.config.timelimit / 3600

    @property
    def cpuh_utilization(self) -> float:
        """ Total active runtime of all workers. """
        return sum([allocation[0] for allocation in self.assignment.values()]) * self.config.cpus_per_worker / 3600

    def schedule_work(self, profile: pd.DataFrame, runtime_estimates: pd.Series) -> pd.DataFrame:
        """ Attempts to greedily fill up all available workers within this job allocation and returns the part of the
        input profile that could not be fit within this job allocation. """

        assert profile.index.isin(runtime_estimates.index).all(), "Mismatch between the index of the given runtime " \
                                                                  "estimates and the profile."
        job_config = self.config
        workers = self.workers
        assignment = self.assignment
        worker_id_cycle = itertools.cycle(workers)
        curr_worker = next(worker_id_cycle)

        def find_next_eligible_worker(required_runtime: float) -> Optional[WorkerConfig]:
            """ Greedily fill up each worker's budget before moving on to the next worker. """
            nonlocal curr_worker
            counter = itertools.count()
            while next(counter) < job_config.workers_per_job:
                available_runtime = job_config.timelimit - assignment[curr_worker][0]
                if available_runtime >= required_runtime:
                    return curr_worker
                else:
                    curr_worker = next(worker_id_cycle)
            return None

        remainder = []
        for conf in profile.index:
            runtime = runtime_estimates[conf]
            assigned_worker = find_next_eligible_worker(required_runtime=runtime)
            if assigned_worker is None:
                remainder.append(conf)
                continue

            assignment[assigned_worker][0] += runtime
            assignment[assigned_worker][1].append(conf)

        for worker in workers:
            basedirs: pd.Series = profile.loc[assignment[worker][1], ("job_config", "basedir")].rename("basedir")
            basedirs.sort_index(axis=0, inplace=True)
            worker.portfolio = basedirs

        # final_assignment = {k: tuple(v) for k, v in assignment.items()}
        remainder = profile.loc[remainder, :]

        return remainder

    def optimize(self):
        """ If the flags 'dynamic_timelimit' and 'dynamic_nnodes' are True, optimize the job's resource demands w.r.t.
        the allocated work so as to maximize the CPUh utilization. """

        _log.debug(f"Current CPUh utilization: {self.cpuh_utilization * 100 / self.cpuh_demand:.2f}%")
        job_config = self.config

        if self.dynamic_nnodes:
            # If any workers don't have any work assigned to them, don't request those nodes in the first place.
            empty_workers = []
            for worker, (runtime, configs) in self.assignment.items():
                if runtime == 0.:
                    empty_workers.append(worker)

            for worker in empty_workers:
                self.assignment.pop(worker)

            self.workers = list(self.assignment.keys())

            empty_cpus = len(empty_workers) * job_config.cpus_per_worker
            empty_nodes = empty_cpus // job_config.cpus_per_node
            if empty_nodes > 0:
                job_config.nodes_per_job -= empty_nodes
                _log.debug(f"Reduced number of nodes required by {empty_nodes}.New CPUh utilization is at "
                           f"{self.cpuh_utilization * 100 / self.cpuh_demand:.2f}%")

        if self.dynamic_timelimit:
            # Clip the job runtime to the maximum required by any worker.
            max_runtime = max([allocation[0] for allocation in self.assignment.values()])
            job_config.timelimit = max_runtime
            _log.debug(f"Re-adjusted the job length to (DD-HH:MM): {JobConfig.timestr(job_config.timelimit)}. New CPUh "
                       f"utilization is at {self.cpuh_utilization * 100 / self.cpuh_demand:.2f}%")

    def save_worker_portfolios(self, jobid: int, portfolio_dir: Path) -> Path:
        """ Saves the respective portfolios for each worker in the given directory 'portfolio_dir'. The directory
        is created if it doesn't already exist. """

        portfolio_dir.mkdir(exist_ok=True, parents=False)
        nworkers = len(self.workers)
        portfolios = {w.worker_id: w.portfolio for w in self.workers}
        job_portfolio = pd.concat(portfolios, axis=0)
        # TODO: Save index level name in constants
        job_portfolio.index = job_portfolio.index.set_names("worker_id", level=0)
        portfolio_fn = portfolio_dir / f"portfolio_{jobid}.pkl.gz"
        job_portfolio.to_pickle(portfolio_fn)
        return portfolio_fn
        # for i, worker in enumerate(self.workers, start=1):
        #     worker.portfolio_dir = portfolio_dir
        #     worker.portfolio.to_pickle(worker.portolio_file)
        #
        #     if (i % (5 * max(1, nworkers // 100))) == 0:
        #         _log.info(f"Saved {i / nworkers} portfolios ({i * 100 / nworkers:.2f}%).")


def fidelity_basedir_map(c: Union[pd.Series, dict]):
    """ Given a Series or mapping from fidelity parameter names to values, generate a fidelity-specific base directory
    name. """

    global fidelity_params, fidelity_types
    return "-".join([f"{p}-{fidelity_types[p](c[i])}" for i, p in enumerate(fidelity_params)]) + "/tasks"


# noinspection PyPep8
def allocate_work(job_config: JobConfig, profile: pd.DataFrame, epochs: int, cpuh_utilization_cutoff: float = 0.75,
                  dynamic_timelimit: bool = True, dynamic_nnodes: bool = True, remainder_pth: Path = None,
                  worker_id_offset: int = 0, squish_configs: bool = False) -> Sequence[JobAllocation]:
    """
    Given a job configuration with estimates of each model's required runtime per epoch, generates a work schedule in
    the form of a list of WorkerConfig objects that have been allocated their respective portfolios.
    'runtime_estimates' should be a pandas DataFrame with a MultiIndex index containing model IDs as defined by the
    unique tuple (<taskid>, <model_idx>), and MultiIndex columns with values [('model_config', <conf>)] and
    [('runtime', 'required')], where <conf> represents all the names of the parameters in the config.
    Scheduling is performed using a not very efficient O(m.n) greedy solver, where m is the number of configurations
    to be scheduled and n is the number of workers available.
    """

    per_epoch_runtimes: pd.Series = profile.required.runtime

    if ("status", "nepochs") in profile.columns:
        remaining_epochs: pd.Series = epochs - profile.status.nepochs
    else:
        remaining_epochs: int = epochs

    runtime_estimates: pd.Series = per_epoch_runtimes * remaining_epochs
    runtime_estimates = runtime_estimates.sort_values(axis=0, ascending=False)

    sel = runtime_estimates > 0.
    if sel.index.size != profile.index.size:
        _log.info(f"Filtering out {profile.index.size - sel.index.size} negative values for required runtime.")

    runtime_estimates = runtime_estimates[sel]
    profile = profile.loc[runtime_estimates.index]

    sel = runtime_estimates.notna()
    if sel.index.size != profile.index.size:
        _log.info(f"Filtering out {profile.index.size - sel.index.size} NaN values for required runtime.")

    runtime_estimates = runtime_estimates[sel]
    profile = profile.loc[runtime_estimates.index]

    if squish_configs:
        squishable = runtime_estimates[runtime_estimates > job_config.timelimit]
        _log.info(f"Attempting to squish the {squishable.index.size} extra long configs into the given job "
                  f"resource allocation.")
        runtime_estimates[squishable.index] = job_config.timelimit

    total_required_budget = runtime_estimates.sum()

    _log.info(f"Estimated total CPUh requirement: "
              f"{total_required_budget * job_config.cpus_per_worker / 3600:<,.2f}")

    _log.info(f"Generating work schedule for {runtime_estimates.size} configurations spread across "
              f"{job_config.workers_per_job} workers per job.")

    # workers = [WorkerConfig(worker_id=i) for i in range(available_num_workers)]
    # work = {w: [job_config.timelimit, []] for w in workers}
    #
    # worker_id_cycle = itertools.cycle(workers)
    #
    # def find_next_eligible_worker(required_runtime: float) -> Optional[WorkerConfig]:
    #     counter = itertools.count()
    #     while next(counter) < available_num_workers:
    #         curr_worker = next(worker_id_cycle)
    #         available_runtime = work[curr_worker][0]
    #         if available_runtime >= required_runtime:
    #             return curr_worker
    #     return None
    #
    # for conf in runtime_estimates.index:
    #     runtime = runtime_estimates[conf]
    #     assigned_worker = find_next_eligible_worker(required_runtime=runtime)
    #     if assigned_worker is None:
    #         _log.info(f"Found no available work slots for config id {conf}, required runtime - "
    #                   f"{JobConfig.timestr(runtime)}")
    #         continue
    #
    #     work[assigned_worker][0] -= runtime
    #     work[assigned_worker][1].append(conf)
    #
    # for worker in workers:
    #     basedirs: pd.Series = profile.loc[work[worker][1], ("job_config", "basedir")].rename("basedir")
    #     basedirs.sort_index(axis=0, inplace=True)
    #     worker.portfolio = basedirs

    current_profile = profile
    jobs = []
    while True:
        job_allocation = JobAllocation(config=job_config, worker_id_offset=worker_id_offset,
                                       dynamic_nnodes=dynamic_nnodes, dynamic_timelimit=dynamic_timelimit)
        remainder = job_allocation.schedule_work(profile=current_profile, runtime_estimates=runtime_estimates)

        if remainder.index.size == current_profile.index.size:
            # The job timelimit is too small for these configs, which is why no more work can be assigned

            _log.warning(f"The given job timelimit {JobConfig.timestr(job_config.timelimit)} (DD-HH:MM) is too small "
                         f"for {remainder.index.size} configs with an average runtime requirement of "
                         f"{JobConfig.timestr(runtime_estimates[remainder.index].mean())} (DD-HH:MM). The remaining "
                         f"configs were distributed across {len(jobs)} jobs.")

            if remainder_pth is not None:
                _log.info(f"Saving {remainder.index.size} leftover configs at {remainder_pth}")
                remainder.to_pickle(remainder_pth)

            break
        else:
            # A job allocation has been successful. Add it to the list of jobs.
            job_allocation.optimize()
            jobs.append(job_allocation)
            worker_id_offset += job_allocation.config.workers_per_job
            if remainder.index.size == 0:
                _log.info(
                    f"Distributed the workload across {len(jobs)} jobs.")
                break
            else:
                current_profile = remainder

    utilization = [job.cpuh_utilization for job in jobs]
    demand = [job.cpuh_demand for job in jobs]
    avg_utilization = sum(utilization) / sum(demand)
    utilization = [u / d for u, d in zip(utilization, demand)]
    underutilized = avg_utilization  < cpuh_utilization_cutoff

    nworkers = sum([j.config.workers_per_job for j in jobs])

    if underutilized:
        _log.warning(f"The current job setup may not utilize all available workers very well. The current setup "
                     f"demands {sum(demand):<,.2f} CPUh and has a predicted approximate CPUh utilization of "
                     f"{avg_utilization * 100:.2f}%. Individual jobs have CPUh utilization in the range of "
                     f"{min(utilization) * 100:.2f}% to {max(utilization) * 100:.2f}%, spread over {nworkers} workers. "
                     f"To continue adding workers to this job allocation, use the worker ID offset {worker_id_offset}.")
    else:
        _log.info(f"The job setup has an overall CPUh utilization factor of {avg_utilization * 100:.2f}%. Individual "
                  f"jobs have CPUh utilization in the range of {min(utilization) * 100:.2f}% to "
                  f"{max(utilization) * 100:.2f}%, spread over {nworkers} workers. To continue adding workers to this "
                  f"job allocation, use the worker ID offset {worker_id_offset}.")

    return jobs


def generate_model_ids(specs: pd.DataFrame):
    """ Generate all expected model ids i.e. (taskid, model_idx), given the specifications in 'specs'. 'specs' should
    be a dataframe with the fidelity values as index and the columns [ntasks, nsamples] as columns. For each fidelity
    value i.e. row, it is assumed that 'ntasks' different task ids with 'nsamples' model_idx values each are needed.
    The fidelity values are then converted into partial model configs for the newly generated model ids. Thus, the
    returned dataframe will have an index corresponding to every (taskid, model_idx) expected to be generated by the
    input specifications. The columns will instead contain the fidelity values corresponding to each model id. """

    global fidelity_params

    assert "ntasks" in specs.columns, "The 'specs' dataframe should contain the column 'ntasks'."
    assert "nsamples" in specs.columns, "The 'specs' dataframe should contain the column 'nsamples'."

    for f in fidelity_params:
        assert f in specs.index.names, f"The fidelity parameter {f} is missing from the index of the specifications " \
                                       f"DataFrame"

    specs = specs.reorder_levels(fidelity_params, axis=0)
    partial_portfolios = []
    taskid_offset = 0

    for f in specs.index:
        ntasks = specs.loc[f]["ntasks"]
        nsamples = specs.loc[f]["nsamples"]
        index = pd.MultiIndex.from_product(
            [list(range(taskid_offset, taskid_offset + ntasks)), list(range(1, 1 + nsamples))],
            names=[constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value]
        )
        df = pd.DataFrame({n: f[i] for i, n in enumerate(fidelity_params)}, index=index, columns=fidelity_params)
        partial_portfolios.append(df)
        taskid_offset += ntasks

    full_portfolio = pd.concat(partial_portfolios, axis=0)
    return full_portfolio


def generate_basedirs(specs: pd.DataFrame) -> pd.Series:
    """ Given a dataframe such that the index values are unique model ids - (taskid, model_idx) tuples - and the
    columns are fidelity values corresponding to that model id, returns a Series "basedir" that contains a string which
    can be suffixed to a common root directory in order to generate a fidelity-wise base directory for a DirectoryTree
    object. """

    assert constants.MetricDFIndexLevels.taskid.value in specs.index.names, \
        f"The input 'specs' DataFrame must contain the index level {constants.MetricDFIndexLevels.taskid.value}"
    assert constants.MetricDFIndexLevels.modelid.value in specs.index.names, \
        f"The input 'specs' DataFrame must contain the index level {constants.MetricDFIndexLevels.modelid.value}"

    # def mapper(fidelity: pd.Series) -> str:
    #     return "-".join([f"{n}-{fidelity[n]}" for n in fidelity.index])

    return specs.apply(fidelity_basedir_map, axis=1).rename("basedir")


def prepare_directory_structure(basedirs: pd.Series, rootdir: Path):
    """ Given a Series 'basedirs' as input, with index consisting of tuples (taskid, model_idx) and directory names
    (strings) as the data values, and a path to a root directory, creates a large directory tree such that all future
    benchmarking operations in this root directory for the model configs identified by the input Series' index can
    assume that all relevant directories already exist. """

    _log.info("Preparing benchmark data directory structure.")

    assert rootdir.exists(), f"The root directory {rootdir} must be created before the prepare function can be called."
    index_levels = [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value]
    logdir = rootdir / logdir_name
    logdir.mkdir(exist_ok=True, parents=False)

    for level in index_levels:
        assert level in basedirs.index.names, f"The input Series must have {level} as an index level."

    basedirs = basedirs.reorder_levels(index_levels)
    nconfigs = basedirs.size

    for i, (taskid, model_idx) in enumerate(basedirs.index, start=1):
        base = rootdir / basedirs[taskid, model_idx]
        base.mkdir(exist_ok=True, parents=True)
        try:
            tree = utils.DirectoryTree(basedir=base, read_only=False)
        except FileExistsError:
            pass

        try:
            tree.taskid = taskid
        except FileExistsError:
            pass

        try:
            tree.model_idx = model_idx
        except FileExistsError:
            pass

        if i % (5 * max(nconfigs // 100, 1)) == 0:
            # Report progress after iterating over approximately every 5 percent of the configs
            _log.info(f"Finished:\t{i * 100 / nconfigs:=6.2f}%\t({i}/{nconfigs}).")

    _log.info("Finished creating the directory structure.")


# TODO: Overhaul the entire profile generation script to instead work on the concept of accepting only expected mean
#  runtime per epoch, cpus per task/fidelity, required number of evaluations per fidelity, required number of epochs
#  and the available resources per job in order to distribute the workload across the workers. ntasks per fidelity
#  should also be a user-specified fixed value whose purpose is not to ensure that workers don't clash but to control
#  the spread of the directory tree as well as the number of parallel bit streams.
def prepare_full_sampling_profile(cpus_per_task_per_bucket: pd.Series,
                                  runtime_per_task_per_bucket: pd.Series, evals_per_task: pd.Series,
                                  nodes_per_bucket: pd.Series):
    """ Given fidelity-wise estimates of various resource requirements, generate a complete sampling profile containing
    the resource requirements of each individual model config - identified by the tuple (taskid, model_idx) - along
    with their respective base directories and partial configs (fidelity values) as a single DataFrame.

    Final DataFrame structure:
    index - taskid, model_idx
    columns - (required, runtime), (required, cpus), [(model_config, <fidelity parameters>)], (job_config, basedir)
    """

    global fidelity_params

    cpus_per_task_per_bucket: pd.Series = cpus_per_task_per_bucket.astype(int)
    evals_per_task: pd.Series = evals_per_task.astype(int)
    nodes_per_bucket: pd.Series = nodes_per_bucket.astype(int)

    # TODO: Fix this. The input should be workers per bucket - nodes per bucket, cpus per bucket, etc. can easily be
    #  inferred from the current resource availability and should be inferred on an ad-hoc basis

    # 48 = cpus per node, this has been fixed because this is the underlying assumption for the calculations behind
    # the estimates.
    ntasks_per_node = cpus_per_task_per_bucket.rfloordiv(48)
    total_ntasks = ntasks_per_node * nodes_per_bucket
    model_ids = generate_model_ids(
        specs=pd.concat([total_ntasks.rename("ntasks"), evals_per_task.rename("nsamples")], axis=1)
    )
    basedirs = generate_basedirs(specs=model_ids)

    # This is a change in philosophy for how the jobs should be built-up going forward as opposed to what was done thus
    # far. In a future version, this hack for converting runtimes back to runtimes per epoch won't be necessary.
    # noinspection PyArgumentList
    runtime_per_evaluation_per_epoch = runtime_per_task_per_bucket.div(evals_per_task) / 200

    required_runtime = model_ids.join(runtime_per_evaluation_per_epoch.rename("runtime"), on=fidelity_params)["runtime"]
    required_cpus = model_ids.join(cpus_per_task_per_bucket.rename("cpus"), on=fidelity_params)["cpus"]
    profile = pd.concat(
        {
            "required": pd.concat([required_runtime, required_cpus], axis=1),
            "model_config": model_ids,
            "job_config": basedirs
        },
        axis=1
    )

    return profile


if __name__ == "__main__":
    # test script
    # TODO: Adjust for new sampling-profile based operation
    # import sys
    #
    # fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(fmt)
    # _log.addHandler(ch)
    # _log.setLevel(logging.DEBUG)
    #
    # modelid_level_names = [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value]
    # modelids = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2]], names=modelid_level_names)
    # runtimes = [100] * 9
    # nepochs = [300, 200, 200, 100, 100, 50, 50, 25, 25]
    # estimates = {}
    # max_epochs = 200
    #
    # subdfs = {}
    # for id, e, t in zip(modelids.values, nepochs, runtimes):
    #     tdata = [i * t / (e - 1) for i in range(e)]
    #     edata = [0.1 * t / e] * e
    #     subdfs[id] = pd.DataFrame({("diagnostic", "runtime"): tdata, ("train", "duration"): edata,
    #                                ("valid", "duration"): edata, ("test", "duration"): edata})
    #     estimates[id] = t / e * (max_epochs - e)
    #
    # metdf = pd.concat(subdfs, axis=0)
    # metdf.index = metdf.index.rename([*modelid_level_names, constants.MetricDFIndexLevels.epoch.value])
    # runtime_estimates = estimate_remaining_runtime(df=metdf, max_epochs=max_epochs)
    # assert runtime_estimates.columns.size == 1 and "required" in runtime_estimates.columns, \
    #     f"Unexpected runtime estimate DataFrame structure, with columns: {runtime_estimates.columns}"
    # assert runtime_estimates.index.size == len(modelids) and runtime_estimates.index.difference(modelids).size == 0, \
    #     f"Unexpected runtime estimate DataFrame structure.\nExpected model IDs: {modelids}\n" \
    #     f"DataFrame index:{runtime_estimates.index}"
    #
    # expected_estimates = pd.DataFrame(data=estimates, index=["required"]).transpose()
    # assert runtime_estimates.equals(expected_estimates), \
    #     f"Mismatch in expected runtime estimates and calculated estimates.\nExpected: {expected_estimates}\n" \
    #     f"Calculated: {runtime_estimates}"
    #
    # max_runtime = 700.
    # real_fidelity_params = fidelity_params
    # real_fidelity_types = fidelity_types
    # fidelity_params = ["C1", "C2"]
    # fidelity_types = {"C1": int, "C2": int}
    #
    # confs = {"C1": list(range(modelids.size)), "C2": list(range(modelids.size))[::-1]}
    # confs = pd.DataFrame(confs, index=modelids)
    # estimates_input_df = pd.concat({"model_config": confs, "runtime": runtime_estimates}, axis=1)
    # jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=300)
    # work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
    #                                 cpuh_utilization_cutoff=0.75, dynamic_timelimit=False)
    # _log.info("This message should have been preceeded by a warning about low CPUh utilization, at about 0.33.")
    # assert len(work_allocation) == 9, f"Expected the work allocation to be split amongst 9 workers, was split " \
    #                                   f"amongst {len(work_allocation)} workers instead."
    # assert all([w.portfolio.shape[0] == 0 for w in work_allocation[3:]]), \
    #     f"Unexpected work allocation, work allocation for 6 of the 9 workers should have failed, was instead:\n" \
    #     f"{[str(w.portfolio) for w in work_allocation]}"
    #
    # jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=900)
    # work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
    #                                 cpuh_utilization_cutoff=0.75, dynamic_timelimit=False)
    # _log.info("This message should not have been preceeded by a warning about low CPUh utilization.")
    # assert len(work_allocation) == 3, f"Expected the work allocation to be split amongst 3 workers, was split " \
    #                                   f"amongst {len(work_allocation)} workers instead."
    # assert jobconf.timelimit == 900, f"The timelimit of the job should not have been capped."
    #
    # jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=900)
    # work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
    #                                 cpuh_utilization_cutoff=0.95, dynamic_timelimit=True)
    # _log.info("This message should have been preceeded by a warning about low CPUh utilization.")
    # assert len(work_allocation) == 3, f"Expected the work allocation to be split amongst 3 workers, was split " \
    #                                   f"amongst {len(work_allocation)} workers instead."
    # assert jobconf.timelimit == 800, f"The timelimit of the job should have been capped."
    #
    # fidelity_params = real_fidelity_params
    # fidelity_types = real_fidelity_types

    # Test resource estimation funcs

    fids = pd.MultiIndex.from_product([[1, ], [4, ], [0.25, 0.5, 1.0]], names=fidelity_params)
    cpus_per_bucket = pd.Series([1, 1, 4], index=fids)
    runtime_per_bucket = pd.Series([100, 400, 1600], index=fids)
    evals_per_worker = pd.Series([20, 20, 5], index=fids)
    test_nodes_per_bucket = pd.Series([5, 5, 20], index=fids)
    test_profile = prepare_full_sampling_profile(cpus_per_bucket, runtime_per_bucket, evals_per_worker,
                                                 test_nodes_per_bucket)

    assert len(test_profile.index) == 10800, f"Unexpected number of model configs: {len(test_profile.index)}"

    _log.info("Finished verification.")
