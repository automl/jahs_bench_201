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

profiles = profile_data["profiles"]
task_allocations = profile_data["task_allocations"]

completed = []
incomplete = {}
n_incomp = 0

for taskid in basedir.iterdir():
    if not taskid.is_dir():
        continue
    allocations = task_allocations[int(taskid.name)]
    datadir = taskid / "config_data"
    json_files = list(taskid.glob("*.json"))
    n_expected_profiles = allocations[1] - allocations[0]
    for i in range(*allocations):
        json_file = datadir / f"{i}.json"
        if json_file.exists() and json_file.is_file():
            completed.append((i, profiles[i]))
        else:
            if taskid.name not in incomplete:
                incomplete[taskid.name] = []
            incomplete[taskid.name].append((i, profiles[i]))
            n_incomp += 1

results = {
    "completed": completed,
    "incomplete": incomplete
}
with open(basedir / "results.json", "w") as fp:
    json.dump(results, fp, indent=4)

if args.summarize:
    print(f"Task-wise summary of incomplete profile evaluations:\n")
    for taskid in sorted(incomplete.keys(), key=int):
        allocs = task_allocations[int(taskid)]
        print(f"{taskid}: {len(incomplete[taskid])}/{allocs[1] - allocs[0]} incomplete evaluations.")
    print(f"\nFound {len(completed)} completed and {n_incomp} incomplete profiles, out of a total {len(profiles)} expected profiles.")
    print(f"{len(incomplete)}/{len(task_allocations)} tasks are incomplete.")

