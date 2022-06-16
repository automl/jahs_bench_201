import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=Path,
                    help="Path to the base directory where all the tasks' outputs are stored. This script will "
                         "look for JSON files named 'iostats.json' and store the combined stats results in "
                         "basedir/subdir/stats.json. subdir can only be changed in the program code itself.")
parser.add_argument("--plot", action="store_true", help="If specified, also generates the I/O Plot.")
parser.add_argument("--plot_only", action="store_true",
                    help="If specified, looks for a top-level stats.json file and generates a plot directly from "
                         "that data without assembling data from subdirectories. Overrides --plot.")
args = parser.parse_args()

basedir: Path = args.basedir

subdirs = list(map(lambda x: str(x), [1, 4, 16, 64, 256, 512, 1024]))

overall_stats = {}
for sub in subdirs:
    dirpath = basedir / sub

    if args.plot_only:
        with open(dirpath / "stats.json", "r") as fp:
            overall_stats[sub] = json.load(fp)
    else:
        overall_stats[sub] = {
            "init_duration": [],
            "wc_duration": [],
            "proc_duration": [],
        }

        for stats_file in dirpath.rglob("iostats.json"):
            with open(stats_file) as fp:
                iodata = json.load(fp)

            if not isinstance(iodata, dict):
                print(f"Found invalid file structure in {stats_file}. Skipping.")

            try:
                for key, val in iodata.items():
                    overall_stats[sub][key].append(val)
            except KeyError as e:
                print(f"Found invalid file structure in {stats_file} - unknown key {key}. Skipping.")

        with open(dirpath / "stats.json", "w") as fp:
            json.dump(overall_stats[sub], fp)

if args.plot_only or args.plot:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcdefaults()
    plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16})

    fig, ax = plt.subplots(figsize=(8, 4))
    nnodes = np.array(list(map(lambda x: int(x), subdirs)))
    wc_durations: np.ndarray = np.array([np.mean(overall_stats[nn]["wc_duration"]) for nn in subdirs])
    proc_durations: np.ndarray = np.array([np.mean(overall_stats[nn]["proc_duration"]) for nn in subdirs])
    proc_durations_std: np.ndarray = np.array([np.std(overall_stats[nn]["proc_duration"]) for nn in subdirs])

    ax.vlines(nnodes, ymin=0, ymax=100, color="gray", linestyles="-")
    # ax.plot(nnodes, wc_durations, 'bo', linestyle='--', label="Wallclock Duration")
    # ax.plot(nnodes, proc_durations, 'go', linestyle='--', label="Process Duration")
    ax.errorbar(nnodes, proc_durations, yerr=proc_durations_std, linestyle='--', fmt='go', ecolor='red', capsize=5.,
                label="Process Duration")
    ax: plt.Axes
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.set_ylim((1.95, 2.04))

    ax.set_xscale("log")
    ax.set_xlabel("Number of nodes (log scale)")
    ax.set_ylabel("Average read time (seconds)")
    ax.legend()

    for nn in nnodes:
        ax.text(x=nn, y=ax.get_ylim()[0], s=nn, fontsize=10)

    fig.tight_layout()

    # time.sleep(3)

    fig.savefig(basedir / "ioplot.pdf")
    # plt.show()
