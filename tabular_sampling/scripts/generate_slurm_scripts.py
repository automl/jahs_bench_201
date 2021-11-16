from pathlib import Path
import pandas as pd
from string import Template
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--template_dir", type=Path, help="This is the directory where the input templates and "
                                                      "dataframes are stored.")
parser.add_argument("--script_dir", type=Path, help="This is the directory where the generated job scripts will be "
                                                    "stored.")
parser.add_argument("--slurm_dir", type=Path, help="This is the directory where the outputs of the jobs will be "
                                                   "stored i.e. WORK.")
args = parser.parse_args()

pth = args.template_dir
cpus_per_worker_per_node_per_bucket: pd.Series = pd.read_pickle(pth / "cpus_per_worker_per_node_per_bucket.pkl.gz")
cpuh_per_worker_per_bucket: pd.Series = pd.read_pickle(pth / "cpuh_per_worker_per_bucket.pkl.gz")
evals_per_worker: pd.Series = pd.read_pickle(pth / "evals_per_worker.pkl.gz")
nodes_per_bucket: pd.Series = pd.read_pickle(pth / "nodes_per_bucket.pkl.gz")

with open(pth / "job.template") as fp:
    job_template = fp.read()

with open(pth / "config.template") as fp:
    config_template = fp.read()

fids = cpuh_per_worker_per_bucket.index
scriptdir = args.script_dir
scriptdir.mkdir(exist_ok=True, parents=True)

slurm_dir = args.slurm_dir


def timestr(secs: float):
    minutes = math.ceil(secs / 60)
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24
    return f"{days}-{hours:02}:{minutes:02}"


task_offset = 0
for f in fids:
    ncpus = int(cpus_per_worker_per_node_per_bucket[f])
    cpuh = cpuh_per_worker_per_bucket[f]
    cpuh = timestr(cpuh)
    nsamples = int(evals_per_worker[f])
    nnodes = int(nodes_per_bucket[f])
    ntasks = 48 // ncpus

    job_name = f"{'-'.join(['-'.join([n, str(i)]) for n, i in zip(fids.names, f)])}"
    jobdir = slurm_dir / job_name
    jobdir.mkdir(exist_ok=True)
    (jobdir / "tasks").mkdir(exist_ok=True)
    (jobdir / "logs").mkdir(exist_ok=True)

    job_script = Template(job_template).substitute(cpuh=cpuh, nnodes=nnodes, job_name=job_name, ncpus=ncpus,
                                                   ntasks=ntasks, jobdir=jobdir, scriptdir=scriptdir)
    config_script = Template(config_template).substitute(nsamples=nsamples, task_offset=task_offset, **{n: i for n, i in zip(fids.names, f)})

    with open(scriptdir / f"{job_name}.job", "w") as fp:
        fp.write(job_script)

    with open(scriptdir / f"{job_name}.config", "w") as fp:
        fp.write(config_script)

    task_offset += ntasks * nnodes

