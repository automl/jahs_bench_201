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
from typing import Optional, Sequence, Iterator, Tuple

from tabular_sampling.lib import constants
from tabular_sampling.lib.postprocessing.metric_df_ops import estimate_remaining_runtime

_log = logging.getLogger(__name__)


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

    fidelity_params = ["N", "W", "Resolution"]
    def basedir_map(c: pd.Series):
        return "-".join(["-".join([p, str(c[p])]) for p in fidelity_params]) + "/tasks"

    for id, worker in enumerate(workers):
        configs: pd.DataFrame = runtime_estimates.loc[work[worker][1]]["model_config"].copy()
        basedirs: pd.DataFrame = configs.apply(basedir_map, axis=1)
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
    runtime_estimates = estimate_remaining_runtime(metdf, max_epochs=max_epochs)
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

    _log.info("Finished verification.")
