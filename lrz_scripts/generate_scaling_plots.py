import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pth = Path("/home/archit/thesis/scaling_plot_stats")

log_scale_axes = True
# log_scale_axes = False

budgets = {"1.0", "2.0"}
# budgets = {"1.0"}
# budgets = {"2.0"}

nnodes, nconfigs = list(), list()
for d in pth.iterdir():
    stats = d / "stats.json"
    if not stats.exists() or not stats.is_file():
        print(f"Could not find file {stats}.")
        continue
    with open(stats) as fp:
        st = json.load(fp)
    nnodes.append(st["nnodes"])
    nconfigs.append(sum([v for b, v in st["nconfigs"].items() if b in budgets]))

x = np.array(nnodes) # Since each job was run for 5 hours
order = x.argsort()
x = x[order]
y = np.array(nconfigs)[order]

# Dark Mode
# plt.rcParams.update({"axes.facecolor": "black", "figure.facecolor": "black", "axes.labelcolor": "white",
#                      "axes.titlecolor": "white", "axes.edgecolor": "silver", "text.color": "white",
#                      "xtick.color": "white", "ytick.color": "white", "grid.color": "silver"})

# Common settings
plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16})
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax: plt.Axes

ax.plot(x, y, marker="*", markerfacecolor="red", markeredgecolor="red", linewidth=2, markersize=12, linestyle="dashed", color="blue")

if log_scale_axes:
    ymin = np.min(y) / 2
    ymax = np.max(y) * 2
else:
    ymin = np.min(y) * 0.8
    ymax = np.max(y) * 1.2

ax.set_ylim(ymin, ymax)
ax.vlines(x, ymin=ymin, ymax=ymax, color="gray", linestyles="--")

for nn in nnodes:
    ax.text(x=nn, y=ymin, s=nn, fontsize=12)

ax.set_xlabel("# Nodes (5 hours per node)", size=14)
ax.set_ylabel("# Evaluations", size=14)
ax.set_title("Scaling plot for throughput against node-hours spent.", size=16)

if log_scale_axes:
    ax.set_xscale("log")
    ax.set_yscale("log")

ax.grid(which="both")
# plt.show()
fig.tight_layout()
if log_scale_axes:
    fig.savefig("scaling_plot_logscale.pdf")
else:
    fig.savefig("scaling_plot.pdf")
