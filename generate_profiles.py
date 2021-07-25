from naslib.search_spaces import NasBench201SearchSpace
import naslib.utils.utils as naslib_utils
import json
import argparse
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--nconfigs", type=int, default=50,
                    help="Number of configurations to be sampled from the search space.")
parser.add_argument("--seed", type=int, default=2021, help="An integer seed for the RNG.")
parser.add_argument("--dir", type=Path, default=Path(".").expanduser(),
                    help="The directory where the generated configurations and profiles will be stored.")
parser.add_argument("--evals-per-task", type=Path, default=None,
                    help="A JSON file containing a description of how many evaluations each task is expected to "
                         "perform. If None (default) a pre-configured distribution will be used and saved to the "
                         "specified output directory.")

args = parser.parse_args()
seed = args.seed
nconfs = args.nconfigs
outdir = args.dir
evals_per_task_fn = args.evals_per_task

configs = []
space = NasBench201SearchSpace()
naslib_utils.set_seed(seed)

img_sizes = [8, 16, 32]
cell_repeats = [1, 2, 5]
network_widths = [1, 2, 3]

# Calculate how the profile evaluations are to be distributed across all the tasks
if evals_per_task_fn is None:
    evals_per_task = {
        8:  {1: 28, 2: 21, 5: 14},
        16: {1: 14, 2: 9, 5: 4},
        32: {1: 4, 2: 2, 5: 1},
    }
    with open(outdir / "task_load_distribution.json", "w") as fp:
        json.dump(evals_per_task, fp, indent=4)
else:
    with open(evals_per_task_fn) as fp:
        evals_per_task = json.load(fp)

for i in range(nconfs):
    conf = space.clone()
    conf.sample_random_architecture()
    configs.append(conf.get_op_indices())

fidelities = []
for s in img_sizes:
    for n in cell_repeats:
        for w in network_widths:
                fidelities.append((s, n, w))

profiles = []
for f in fidelities:
    for conf in configs:
        profiles.append((f, conf))



evals_per_s_and_n = len(network_widths) * nconfs
task_allocations = []
this_task = 0
consumed_evals = 0
for s in img_sizes:
    for n in cell_repeats:
        task_capacity = evals_per_task[s][n]
        # Assume that any single task will only handle evaluations from a single (s, n) pair's profiles
        # By prior calculation, every (s, n) pair requires multiple tasks.
        required_ntasks = math.ceil(evals_per_s_and_n / task_capacity)
        for i in range(this_task, this_task + required_ntasks):
            if consumed_evals >= len(profiles):
                # We've already assigned all profile evaluations to their respective tasks
                break
            else:
                # There is scope to include more tasks
                task_allocations.append((consumed_evals, consumed_evals + task_capacity))
                consumed_evals += task_capacity
        this_task += required_ntasks


output = {
    "configs": configs,
    "fidelities": fidelities,
    "profiles": profiles,
    "task_allocations": task_allocations
}

with open(outdir / "profiles.json", "w") as fp:
    json.dump(output, fp)