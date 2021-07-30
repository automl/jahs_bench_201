from pathlib import Path
import argparse
import json
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=Path, default=Path().cwd(),
                    help="The base directory where the data to be analyzed is located.")
parser.add_argument("--profiles", type=Path, default=Path().cwd() / "profiles.json",
                    help="JSON file containing the appropriate profile configurations to be validated.")
parser.add_argument("--summarize", action="store_true", help="When given, a summary of the validation results is also displayed.")
args = parser.parse_args()
basedir: Path = args.dir
profiles_fn: Path = args.profiles

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


for taskid in basedir.iterdir():
    if not taskid.is_dir():
        continue
    allocations = task_allocations[int(taskid.name)]
    datadir = taskid / "config_data"
    json_files = list(taskid.glob("*.json"))
    n_expected_profiles = allocations[1] - allocations[0]
    for i in range(*allocations):
        json_file = datadir / f"{i}.json"
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
    print(f"Task-wise summary of incomplete profile evaluations:\n")
    for taskid in sorted(incomplete.keys(), key=int):
        allocs = task_allocations[int(taskid)]
        print(f"{taskid}: {len(incomplete[taskid])}/{allocs[1] - allocs[0]} incomplete evaluations.")
    print(f"\nFound {len(completed)} completed and {n_incomp} incomplete profiles, out of a total {len(profiles)} expected profiles.")
    print(f"{len(incomplete)}/{len(task_allocations)} tasks are incomplete.")

