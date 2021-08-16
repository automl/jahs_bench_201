import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=Path, help="Path to the base directory where all the relevant dataframes are "
                                                 "stored as files named data.pkl.gz.")
args = parser.parse_args()

basedir: Path = args.basedir
subdirs = [1, 4, 16, 64, 256, 512, 1024]
dfs = []

def get_num_configs(df: pd.DataFrame):
    df = df.xs(key=0, level="idx")["train_time"]
    return df.count()

counts = []
for sub in subdirs:
    pth = basedir / str(sub)
    df = pd.read_pickle(pth / "data.pkl.gz")
    counts.append(get_num_configs(df))


# Common settings
plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16})
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax: plt.Axes

x = subdirs
y = counts
ax.plot(x, y, marker="*", markerfacecolor="red", markeredgecolor="red", linewidth=2, markersize=12,
        linestyle="dashed", color="blue")

ymin = np.min(y) / 2
ymax = np.max(y) * 2

ax.set_ylim(ymin, ymax)
ax.vlines(x, ymin=ymin, ymax=ymax, color="gray", linestyles="--")

for nn in subdirs:
    ax.text(x=nn, y=ymin, s=nn, fontsize=12)

ax.set_xlabel("# Nodes (2 hours per node)", size=14)
ax.set_ylabel("# Evaluations", size=14)
ax.set_title("Scaling plot for throughput against node-hours spent.", size=16)

ax.set_xscale("log")
ax.set_yscale("log")

ax.grid(which="both")
# plt.show()
fig.tight_layout()
fig.savefig(basedir / "scaling_plot.pdf")
