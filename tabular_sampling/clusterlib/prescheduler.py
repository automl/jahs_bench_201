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
from typing import Optional, Sequence, Iterator, Tuple, Union

from tabular_sampling.lib.core import constants, utils
from tabular_sampling.lib.postprocessing.metric_df_ops import estimate_remaining_runtime
from tabular_sampling.lib.core import constants

_log = logging.getLogger(__name__)
fidelity_params = constants.fidelity_params
fidelity_types = constants.fidelity_types

class JobConfig(object):
    """ A container for various properties of the cluster that will be used to help compute the work schedule. """

    cpus_per_worker: int
    cpus_per_node: int
    nodes_per_job: int
    timelimit: float
    template_file: Optional[Path]

    def __init__(self, cpus_per_worker = 1, cpus_per_node = 48, nodes_per_job = 3072, timelimit = 2 * 24 * 3600,
                 template_file: Path = None):
        self.cpus_per_worker = cpus_per_worker
        self.cpus_per_node = cpus_per_node
        self.nodes_per_job = nodes_per_job
        self.timelimit = timelimit
        self.template_file = template_file

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
            "cpus_per_worker": self.cpus_per_worker,
            "cpus_per_node": self.cpus_per_node,
            "nodes_per_job": self.nodes_per_job,
            "workers_per_node": self.workers_per_node,
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
    portfolio: Optional[pd.DataFrame]

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
        self.portfolio = pd.read_pickle(self.portolio_file)

    def iterate_portfolio(self, start: int = 0, stop: int = None, step: int = 1, rootdir: Optional[Path] = None) -> \
            Iterator[Tuple[int, int, Path]]:
        """ Returns an iterator over this worker's portfolio, consisting of a list of 3-tuples, each consisting of
        an integer task ID, an integer model index, and a path to the base directory where all the metrics and
        checkpoints of this configuration's previous evaluations are stored. The optional Path-like 'rootdir' can be
        specified to read all the saved base directory paths to be treated as relative to the specified 'rootdir'. """

        subset = self.portfolio[start:stop:step]
        index = subset.index.to_list()
        pths = subset["basedir"].to_list()
        if rootdir is None:
            pths = [Path(p) for p in pths]
        else:
            pths = [(rootdir / p).resolve() for p in pths]

        return iter([(*c, p) for c, p in zip(index, pths)])

    def __hash__(self):
        return hash(self.worker_id)


# class ModelConfig(object):
#     """ Container for various properties relevant to each unique model configuration, as defined by the tuple
#     (<taskid>, <model_idx>) """
#
#     taskid: int
#     model_idx: int
#
#     def __init__(self, taskid: int, model_idx: int):
#         self.taskid = taskid
#         self.model_idx = model_idx
#
#     @classmethod
#     def load_from_portfolio(cls, portfolio: pd.DataFrame, taskid: int, model_idx: int) -> ModelConfig:
#         pass


def fidelity_basedir_map(c: Union[pd.Series, dict]):
    """ Given a Series or mapping from fidelity parameter names to values, generate a fidelity-specific base directory
    name. """

    global fidelity_params, fidelity_types
    return "-".join([f"{p}-{fidelity_types[p](c[i])}" for i, p in enumerate(fidelity_params)]) + "/tasks"


def allocate_work(job_config: JobConfig, runtime_estimates: pd.DataFrame, cpuh_utilization_cutoff: float = 0.75,
                  cap_job_timelimit: bool = True) -> Sequence[WorkerConfig]:
    """ Given a job configuration and estimates for how long each model needs to run for, generates a work schedule in
    the form of a list of WorkerConfig objects that have been allocated their respective portfolios.
    'runtime_estimates' should be a pandas DataFrame with a MultiIndex index containing model IDs as defined by the
    unique tuple (<taskid>, <model_idx>), and MultiIndex columns with values [('model_config', <conf>)] and
    [('runtime', 'required')], where <conf> represents all the names of the parameters in the config.
    Scheduling is performed using a not very efficient O(m.n) greedy solver, where m is the number of configurations
    to be scheduled and n is the number of workers available.
    """

    runtime_estimates = runtime_estimates.sort_values(by=("runtime", "required"), axis=0, ascending=False)
    estimates_series = runtime_estimates[("runtime", "required")]

    _log.info("Filtering out negative values for required runtime.")
    sel = estimates_series.where(estimates_series > 0.).notna()
    estimates_series = estimates_series[sel]
    runtime_estimates = runtime_estimates[sel]

    _log.info("Filtering out NA values from runtime estimates.")
    sel = estimates_series.notna()
    estimates_series = estimates_series[sel]
    runtime_estimates = runtime_estimates[sel]

    total_required_budget = estimates_series.sum()
    required_num_workers = total_required_budget // job_config.timelimit + \
                           int(bool(total_required_budget % job_config.timelimit))
    required_num_jobs = math.ceil(required_num_workers / job_config.workers_per_job)

    assert required_num_jobs >= 1, "Could not find a valid allocation under the given constraints. Consider allowing " \
                                   "jobs to be underutilized or changing the job config."

    _log.info(f"Generating work schedule for {estimates_series.size} configurations spread across "
              f"{job_config.workers_per_job} workers per job.")

    workers = [WorkerConfig(worker_id=i) for i in range(job_config.workers_per_job * required_num_jobs)]
    work = {w: [job_config.timelimit, []] for w in workers}
    nworkers = len(workers)

    worker_id_cycle = itertools.cycle(list(range(job_config.workers_per_job)))

    def find_next_eligible_worker(required_runtime: float) -> Optional[WorkerConfig]:
        counter = itertools.count()
        while next(counter) < job_config.workers_per_job:
            curr_worker = workers[next(worker_id_cycle)]
            available_runtime = work[curr_worker][0]
            if available_runtime >= required_runtime:
                return curr_worker
        return None

    for conf, runtime in zip(estimates_series.index.values, estimates_series.values):
        assigned_worker = find_next_eligible_worker(required_runtime=runtime)
        if assigned_worker is None:
            _log.info(f"Found no available work slots for config id {conf}, required runtime - "
                      f"{JobConfig.timestr(runtime)}, "
                      f"configuration: {runtime_estimates.loc[conf]['model_config']} ")
            continue

        work[assigned_worker][0] -= runtime
        work[assigned_worker][1].append(conf)

    for id, worker in enumerate(workers):
        configs: pd.DataFrame = runtime_estimates.loc[work[worker][1]]["model_config"].copy()
        basedirs: pd.DataFrame = configs.apply(fidelity_basedir_map, axis=1)
        if isinstance(basedirs, pd.Series):
            # This is the normal route; only in the case when #workers > #configs is 'basedirs' occasionally an empty
            # DataFrame and not a Series
            basedirs = basedirs.to_frame("basedir")
        basedirs.sort_index(axis=0, inplace=True)
        worker.portfolio = basedirs
        # worker.portfolio = runtime_estimates.loc[work[worker][1]][["model_config"]].copy()
        # worker.portfolio.sort_index(axis=0, inplace=True)

    min_waste = 0.
    if cap_job_timelimit:
        min_waste = min([work[w][0] for w in workers])
        job_config.timelimit = job_config.timelimit - min_waste

    utilization = sum([job_config.timelimit + min_waste - work[w][0] for w in workers])
    utilization /= job_config.timelimit * nworkers
    underutilized = utilization < cpuh_utilization_cutoff

    if underutilized:
        _log.warning(f"The current job setup may not utilize all available workers very well. Current setup predicted "
                     f"to require {required_num_jobs} jobs with a predicted approximate CPUh utilization of "
                     f"{utilization * 100:.2f}%.")
    else:
        _log.debug(f"The job setup has a CPUh utilization factor of {utilization * 100:.2f}%.")

    return workers


def save_worker_portfolios(workers: Sequence[WorkerConfig], portfolio_dir: Path):
    """ Convenience function. Given a list of WorkerConfig objects that have been pre-allocated worker IDs and
    portfolios, saves the respective portfolios for each worker in the given directory 'portfolio_dir'. The directory
    is created if it doesn't already exist. """

    portfolio_dir.mkdir(exist_ok=True, parents=False)
    for worker in workers:
        worker.portfolio_dir = portfolio_dir
        worker.portfolio.to_pickle(worker.portolio_file)


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

    for l in index_levels:
        assert l in basedirs.index.names, f"The input Series must have {l} as an index level."

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

        if i % (5 * nconfigs // 100) == 0:
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

    import sys

    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    _log.addHandler(ch)
    _log.setLevel(logging.DEBUG)

    modelid_level_names = [constants.MetricDFIndexLevels.taskid.value, constants.MetricDFIndexLevels.modelid.value]
    modelids = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2]], names=modelid_level_names)
    runtimes = [100] * 9
    nepochs = [300, 200, 200, 100, 100, 50, 50, 25, 25]
    estimates = {}
    max_epochs = 200

    subdfs = {}
    for id, e, t in zip(modelids.values, nepochs, runtimes):
        tdata = [i * t / (e - 1) for i in range(e)]
        edata = [0.1 * t / e] * e
        subdfs[id] = pd.DataFrame({("diagnostic", "runtime"): tdata, ("train", "duration"): edata,
                                   ("valid", "duration"): edata, ("test", "duration"): edata})
        estimates[id] = t / e * (max_epochs - e)

    metdf = pd.concat(subdfs, axis=0)
    metdf.index = metdf.index.rename([*modelid_level_names, constants.MetricDFIndexLevels.epoch.value])
    runtime_estimates = estimate_remaining_runtime(df=metdf, max_epochs=max_epochs)
    assert runtime_estimates.columns.size == 1 and "required" in runtime_estimates.columns, \
        f"Unexpected runtime estimate DataFrame structure, with columns: {runtime_estimates.columns}"
    assert runtime_estimates.index.size == len(modelids) and runtime_estimates.index.difference(modelids).size == 0, \
        f"Unexpected runtime estimate DataFrame structure.\nExpected model IDs: {modelids}\n" \
        f"DataFrame index:{runtime_estimates.index}"

    expected_estimates = pd.DataFrame(data=estimates, index=["required"]).transpose()
    assert runtime_estimates.equals(expected_estimates), \
        f"Mismatch in expected runtime estimates and calculated estimates.\nExpected: {expected_estimates}\n" \
        f"Calculated: {runtime_estimates}"

    max_runtime = 700.
    real_fidelity_params = fidelity_params
    real_fidelity_types = fidelity_types
    fidelity_params = ["C1", "C2"]
    fidelity_types = {"C1": int, "C2": int}

    confs = {"C1": list(range(modelids.size)), "C2": list(range(modelids.size))[::-1]}
    confs = pd.DataFrame(confs, index=modelids)
    estimates_input_df = pd.concat({"model_config": confs, "runtime": runtime_estimates}, axis=1)
    jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=300)
    work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
                                    cpuh_utilization_cutoff=0.75, cap_job_timelimit=False)
    _log.info("This message should have been preceeded by a warning about low CPUh utilization, at about 0.33.")
    assert len(work_allocation) == 9, f"Expected the work allocation to be split amongst 9 workers, was split " \
                                      f"amongst {len(work_allocation)} workers instead."
    assert all([w.portfolio.shape[0] == 0 for w in work_allocation[3:]]), \
        f"Unexpected work allocation, work allocation for 6 of the 9 workers should have failed, was instead:\n" \
        f"{[str(w.portfolio) for w in work_allocation]}"

    jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=900)
    work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
                                    cpuh_utilization_cutoff=0.75, cap_job_timelimit=False)
    _log.info("This message should not have been preceeded by a warning about low CPUh utilization.")
    assert len(work_allocation) == 3, f"Expected the work allocation to be split amongst 3 workers, was split " \
                                      f"amongst {len(work_allocation)} workers instead."
    assert jobconf.timelimit == 900, f"The timelimit of the job should not have been capped."

    jobconf = JobConfig(cpus_per_worker=1, cpus_per_node=3, nodes_per_job=1, timelimit=900)
    work_allocation = allocate_work(job_config=jobconf, runtime_estimates=estimates_input_df,
                                    cpuh_utilization_cutoff=0.95, cap_job_timelimit=True)
    _log.info("This message should have been preceeded by a warning about low CPUh utilization.")
    assert len(work_allocation) == 3, f"Expected the work allocation to be split amongst 3 workers, was split " \
                                      f"amongst {len(work_allocation)} workers instead."
    assert jobconf.timelimit == 800, f"The timelimit of the job should have been capped."

    fidelity_params = real_fidelity_params
    fidelity_types = real_fidelity_types

    # Test resource estimation funcs

    fids = pd.MultiIndex.from_product([[1, ], [4, ], [0.25, 0.5, 1.0]], names=fidelity_params)
    cpus_per_bucket = pd.Series([1, 1, 4], index=fids)
    runtime_per_bucket = pd.Series([100, 400, 1600], index=fids)
    evals_per_worker = pd.Series([20, 20, 5], index=fids)
    nodes_per_bucket = pd.Series([5, 5, 20], index=fids)
    profile = prepare_full_sampling_profile(cpus_per_bucket, runtime_per_bucket, evals_per_worker, nodes_per_bucket)

    assert len(profile.index) == 10800, f"Unexpected number of model configs: {len(profile.index)}"

    _log.info("Finished verification.")
