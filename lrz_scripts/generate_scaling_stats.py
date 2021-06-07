import hpbandster.core.result as hpbres
from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser("")
parser.add_argument("--base", type=Path, help="The base directory, sub-directories of which are suppoesed to be "
                                              "individual jobs for which stats are to be collected. The final directory "
                                              "should be parseable by HPBandster.core.result. Expected structure for "
                                              "final directory: <base>/<job_id>/logs/0/run_<run_id>")
args = parser.parse_args()

for dir in args.base.iterdir():
    if not dir.is_dir():
        continue
    pth = Path(dir) / "logs" / "0"

    nnodes = 0
    nconfigs = dict() # Store a counter for number of successfully evaluated configs for each budget
    for subdir in pth.iterdir():
        if not subdir.is_dir():
            continue

        res: hpbres.Result = hpbres.logged_results_to_HBS_result(subdir)
        runs = list(filter(lambda r: r.loss is not None and r.loss < 100000, res.get_all_runs()))
        for r in runs:
            if r.budget not in nconfigs:
                nconfigs[r.budget] = 0
            nconfigs[r.budget] += 1

        nnodes += 1

    json_stats = {
        "nnodes": nnodes,
        "nconfigs": nconfigs
    }

    with open(pth / "stats.json", "w") as fp:
        json.dump(json_stats, fp, indent=4)
