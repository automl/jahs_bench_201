from pathlib import Path
import numpy as np
import pandas as pd
from jahs_bench.tabular.clusterlib.prescheduler import JobConfig

required_evals_per_bucket = 10_000
workers_per_bucket_cap = required_evals_per_bucket
epochs_per_eval = 200
cpus_per_node = 48
total_num_nodes = 6400
available_num_cpus = cpus_per_node * total_num_nodes
max_evals_per_worker = 5

# Assuming linear memory scaling w.r.t N and W, quadratic scaling with Resolution
# For starters, it should suffice to ignore N and W and only scale according to Resolution
# TODO: Come back and revise this if something breaks.

fids = pd.MultiIndex.from_product([[1, 3, 5], [4, 8, 16], [0.25, 0.5, 1.0]], names=["N", "W", "Resolution"])
# fids = pd.MultiIndex.from_product([[1, 3, 5], [1, 2, 3], [8, 16, 32]], names=["N", "W", "Resolution"])
max_fid = (5, 16, 1.0)
max_memory_req = 6  # GB


def approximate_memory_reqs():
    req = pd.DataFrame(index=fids)
    req = req.assign(MemoryReq=max_memory_req)

    scales = 2 ** (2 * np.log2(req.index.get_level_values("Resolution").values)).reshape(-1, 1)
    req = req.mul(scales)
    return req


def final_calculations(runtimes_per_epoch):
    r = approximate_memory_reqs()
    # CPUs per node = memory_required / memory_per_cpu
    cpus_per_worker_per_node_per_bucket: pd.Series = (r / 1.5).clip(1).astype(int)["MemoryReq"]
    max_nodes_per_bucket = cpus_per_worker_per_node_per_bucket.rdiv(cpus_per_node).rdiv(workers_per_bucket_cap)
    max_nodes_per_bucket: pd.Series = max_nodes_per_bucket.clip(upper=total_num_nodes)
    workers_per_bucket = cpus_per_worker_per_node_per_bucket.rdiv(cpus_per_node) * max_nodes_per_bucket
    workers_per_bucket: pd.Series = workers_per_bucket.round()

    inv_time_ratios = runtimes_per_epoch.rdiv(runtimes_per_epoch.min())
    evals_per_worker = inv_time_ratios * max_evals_per_worker
    evals_per_worker = evals_per_worker.round().clip(lower=1).astype(int)
    workers_per_bucket /= evals_per_worker
    nodes_per_bucket = workers_per_bucket * cpus_per_worker_per_node_per_bucket / cpus_per_node
    nodes_per_bucket = nodes_per_bucket.round().astype(int)


    runtime_per_evaluation_per_bucket = runtimes_per_epoch * epochs_per_eval
    cpuh_required_per_bucket = runtime_per_evaluation_per_bucket * required_evals_per_bucket
    runtime_per_worker_per_bucket = cpuh_required_per_bucket / workers_per_bucket

    return cpus_per_worker_per_node_per_bucket, runtime_per_worker_per_bucket, evals_per_worker, nodes_per_bucket


def create_slurm_limits_dataframe(job_config: JobConfig, cpus_per_worker_per_node_per_bucket: pd.Series,
                                  cpuh_per_worker_per_bucket: pd.Series, evals_per_worker: pd.Series,
                                  nodes_per_bucket: pd.Series, worker_id_offset: int = 0,
                                  max_job_runtime: int = 24 * 60 * 60) -> pd.DataFrame:
    """ Use the fidelity-wise distribution of the resources required for some evaluation profile as well as a
    definition of the resources available per job as input in order to bundle up the total computation into the jobs
    that conform to the specified job resource profile. The bundled up jobs are described in the returned DataFrame as
    columns corresponding to resource requirements and indices corresponding to individual jobs and configurations,
    i.e. the index is a MultiIndex (job_id, config_id). It is assumed that all inputs have already been filtered
    appropriately such that there is no need to re-check the demand and supply of certain resources that are fixed
    across multiple bundled jobs, e.g. cpus per task. The resource requirements specified in the input should all be
    pandas Series with matching indices, such that each index value corresponds to a single fidelity bucket. The int
    'worker_id_offset' can be used to specify that these evaluations should be assumed to be continue a separate batch
    of evaluations, such that the last evalatuion in that batch had the worker id 'worker_id_offset - 1', or in other
    words 'worker_id_offset' evaluations have already been accounted for before this batch of evaluations.

    output dataframe format:
    index: MultiIndex, names = [job_id, config_id]
    columns: Index, values = [worker_id_start, worker_id_end, worker_id_offset,

    """

    cpus_per_worker_per_node_per_bucket: pd.Series = cpus_per_worker_per_node_per_bucket.astype(int)  # cpus per task
    evals_per_worker: pd.Series = evals_per_worker.astype(int)
    nodes_per_bucket: pd.Series = nodes_per_bucket.astype(int)

    ntasks_per_node = cpus_per_worker_per_node_per_bucket.rfloordiv(job_config.cpus_per_node)
    total_ntasks = (ntasks_per_node * nodes_per_bucket).sum()
    cpuh_required = cpuh_per_worker_per_bucket.sum() * total_ntasks




if __name__ == "__main__":
    cifar_total_num = 60_000
    colorectal_total_num = 5000
    ucmerced_total_num = 2100
    batch_size = 256

    runtimes_pth = Path("/home/archit/thesis/experiments/checkpointing/tables")
    cifar_runtimes_per_epoch: pd.Series = pd.read_pickle(runtimes_pth / "per_epoch_runtimes.pkl.gz")["mean"]

    ## CIFAR-10
    # outdir = Path("/home/archit/thesis/slurm_configs/cifar10")
    # runtimes_per_epoch = pd.read_pickle(runtimes_pth)["mean"]

    ## COLORECTAL_HISTOLOGY
    # outdir = Path("/home/archit/thesis/slurm_configs/colorectal_histology")
    # runtimes_pth = Path("/home/archit/thesis/slurm_configs/colorectal_histology/")
    # runtimes_per_epoch = pd.read_pickle(runtimes_pth / "approx_runtimes.pkl.gz")
    # memory_ratio = (5_000 / 50_000) * (128**2 / 32 ** 2)
    # max_memory_req = 5 * memory_ratio  # 5 GB base
    #
    # if "resolution" in runtimes_per_epoch.index.names:
    #     runtimes_per_epoch.index = runtimes_per_epoch.index.rename(runtimes_per_epoch.index.names[:2] + ["Resolution"])

    ## UC_MERCED
    outdir = Path("/home/archit/thesis/slurm_configs/uc_merced")
    # runtimes_pth = Path("/home/archit/thesis/slurm_configs/uc_merced/")
    img_size_ratio = 128**2 / 32 ** 2
    nbatches_ratio = (ucmerced_total_num // batch_size) / (cifar_total_num // batch_size)
    nimgs_ratio = ucmerced_total_num / cifar_total_num
    runtime_ratio = nbatches_ratio * img_size_ratio
    memory_ratio = nimgs_ratio * img_size_ratio

    runtimes_per_epoch = cifar_runtimes_per_epoch * runtime_ratio
    runtimes_per_epoch.to_pickle(outdir / "per_epoch_runtimes.pkl.gz")
    max_memory_req = 5 * memory_ratio  # 5 GB base

    if "resolution" in runtimes_per_epoch.index.names:
        runtimes_per_epoch.index = runtimes_per_epoch.index.rename(runtimes_per_epoch.index.names[:2] + ["Resolution"])



    # Map fidelity values to updated ones
    runtimes_per_epoch = runtimes_per_epoch[runtimes_per_epoch.index.get_level_values("N").isin([1, 3, 5])]
    map_W = {1: 4, 2: 8, 3: 16}
    map_Res = {8: 0.25, 16: 0.5, 32: 1.0}
    runtimes_per_epoch.index = runtimes_per_epoch.index.map(lambda x: (x[0], map_W[x[1]], map_Res[x[2]]))

    cpus_per_worker_per_node_per_bucket, runtime_per_worker_per_bucket, evals_per_worker, nodes_per_bucket = \
        final_calculations(runtimes_per_epoch)

    # These are all over the place. This entire script should be phased out in favor of a better profiler that
    # distributes the entire workload across multiple jobs of a given, fixed resource size instead of trying to infer
    # optimal resource allocations from the estimates. The difference in name between the variables and their pickle
    # filenames has been intentionally left in in order to indicate differences in behaviour and interpretation going
    # forward.
    cpus_per_worker_per_node_per_bucket.to_pickle(outdir / "cpus_per_worker_per_node_per_bucket.pkl.gz")
    runtime_per_worker_per_bucket.to_pickle(outdir / "cpuh_per_worker_per_bucket.pkl.gz")
    evals_per_worker.to_pickle(outdir / "evals_per_worker.pkl.gz")
    nodes_per_bucket.to_pickle(outdir / "nodes_per_bucket.pkl.gz")
    pass
