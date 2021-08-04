import logging
from pathlib import Path
import argparse
import json
import numpy
import math

parser = argparse.ArgumentParser()
parser.add_argument("--base", type=Path, default=Path().cwd(), 
                    help="Base directory which contains the relevant 'profiles.json' file and the sub-directory "
                         "'data'.")
parser.add_argument("--summarize", action="store_true", 
                    help="When given, a summary of the validation results is also displayed.")
parser.add_argument("--summarize-to-file", action="store_true", help="When given along with --summarize, stores the "
                                                                     "summary alongside the stats.")
parser.add_argument("--nnodes", type=int, default=0, help="Only used when --summarize is given. Number of nodes used "
                                                          "by the job that generated the data in question.")
parser.add_argument("--budget", type=float, default=0., help="Only used when --summarize is given. Time budget of each "
                                                          "node in hours.")
args = parser.parse_args()

basedir: Path = args.base
datadir: Path = basedir / "data"
profiles_fn: Path = basedir / "profiles.json"
if not (basedir.exists() and basedir.is_dir()):
    raise RuntimeError(f"Given base directory {basedir} does not exist.")
if not (datadir.exists() and datadir.is_dir()):
    raise RuntimeError(f"Could not find a valid data directory at path {datadir}.")
if not (profiles_fn.exists() and profiles_fn.is_file()):
    raise RuntimeError(f"Could not find a valid profiles directory at path {profiles_fn}.")

with open(profiles_fn) as fp:
    profile_data = json.load(fp)

fidelities: list = profile_data["fidelities"]
profiles = profile_data["profiles"]
task_allocations = profile_data["task_allocations"]

stats = {}
completed = []
incomplete = {}
n_incomp = 0


def handle_completed_profile(profile, data):
    fidelity = profile[0]
    key = fidelities.index(fidelity)
    if key not in stats:
        stats[key] = dict(
            nconfigs = 0,
            final_val_accs = [],
            final_test_accs = [],
            runtimes = [],
            train_times = [],
            model_sizes = []
        )

    st = stats[key]
    st["nconfigs"] += 1
    st["final_val_accs"].append(data["val_acc"][-1])
    st["final_test_accs"].append(data["test_acc"])
    st["runtimes"].append(data["runtime"])
    st["train_times"].append(data["train_time"])
    st["model_sizes"].append(data["params"])
    completed.append((i, profiles[i]))


def handle_incomplete_profile(taskid, profile):
    global n_incomp
    if taskid not in incomplete:
        incomplete[taskid] = []
    incomplete[taskid].append(profile)
    n_incomp += 1


for taskid in datadir.iterdir():
    if not taskid.is_dir():
        continue
    allocations = task_allocations[int(taskid.name)]
    task_data_dir = taskid / "config_data"
    json_files = list(taskid.glob("*.json"))
    n_expected_profiles = allocations[1] - allocations[0]
    for i in range(*allocations):
        json_file = task_data_dir / f"{i}.json"
        profile = profiles[i]
        if json_file.exists() and json_file.is_file():
            with open(json_file) as fp:
                data = json.load(fp)
            handle_completed_profile(profile, data)
        else:
            handle_incomplete_profile(taskid.name, profile)

results = {
    "completed": completed,
    "incomplete": incomplete
}

with open(basedir / "valid.json", "w") as fp:
    json.dump(results, fp)

with open(basedir / "stats.json", "w") as fp:
    json.dump(stats, fp)

if args.summarize:
    summary_logger = logging.getLogger("summary")
    fmt = logging.Formatter("%(message)s")
    if args.summarize_to_file:
        handler = logging.FileHandler(basedir / "summary.txt", "w")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    summary_logger.addHandler(handler)
    summary_logger.setLevel(logging.INFO)

    summary_logger.info(f"Task-wise summary of incomplete profile evaluations:\n")
    for taskid in sorted(incomplete.keys(), key=int):
        allocs = task_allocations[int(taskid)]
        summary_logger.info(f"{taskid}: {len(incomplete[taskid])}/{allocs[1] - allocs[0]} incomplete evaluations.")
    summary_logger.info(f"\nFound {len(completed)} completed and {n_incomp} incomplete profiles, out of a total "
                        f"{len(profiles)} expected profiles.")
    summary_logger.info(f"{len(incomplete)}/{len(task_allocations)} tasks are incomplete.")

    nnodes = args.nnodes
    budget = args.budget

    if nnodes <= 0:
        raise RuntimeError(f"Cannot generate the node-wise utilization summary. The number of nodes must be a "
                           f"positive integer, but given {nnodes}.")
    if budget <= 0.:
        raise RuntimeError(f"Cannot generate the node-wise utilzation summary. The budget must be a positive value, "
                           f"but given {budget:.2f}.")

    summary_logger.info(f"\n\nNode-wise budget utilization summary:\n")
    ntasks = float(len(task_allocations))
    tasks_per_node = math.ceil(ntasks / nnodes)
    available_budget = tasks_per_node * budget
    total_utilization = 0.
    for node in range(nnodes):
        utilized_budget = 0.
        for task in range(tasks_per_node):
            task_idx = node * tasks_per_node + task
            if task_idx >= ntasks:
                continue # This should only account for the few empty tasks in the very tail end, i.e. the last node
            allocs = task_allocations[task_idx]
            task_data_dir = datadir / str(task_idx) / "config_data"
            for prof in range(*allocs):
                fn = task_data_dir / f"{prof}.json"
                if not fn.exists():
                    continue
                with open(fn) as fp:
                    evaluation_results = json.load(fp)
                utilized_budget += evaluation_results["runtime"] / 3600
        summary_logger.info(f"{node}: {(utilized_budget * 100. / available_budget):5.2f}%")
        total_utilization += utilized_budget
    summary_logger.info(f"\nAverage node budget utilization: {(total_utilization * 100 / nnodes / available_budget):5.2f}%.")

