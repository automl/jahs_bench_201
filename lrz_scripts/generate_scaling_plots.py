import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pth = Path("/home/archit/thesis/scaling_plot_stats")
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

x = np.array(nnodes) * 5 # Since each job was run for 5 hours
order = x.argsort()
x = x[order]
y = np.array(nconfigs)[order]

# Dark Mode
# plt.rcParams.update({"axes.facecolor": "black", "figure.facecolor": "black", "axes.labelcolor": "white",
#                      "axes.titlecolor": "white", "axes.edgecolor": "silver", "text.color": "white",
#                      "xtick.color": "white", "ytick.color": "white", "grid.color": "silver"})

# Common settings
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12})
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax: plt.Axes

ax.plot(x, y, marker="*", markerfacecolor="red", markeredgecolor="red", linewidth=2, markersize=12, linestyle="dashed", color="blue")
ax.set_xlabel("Node hours", size=14)
ax.set_ylabel("# Evaluations", size=14)
ax.set_title("Scaling plot for throughput against node-hours spent.", size=16)
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_yscale()
ax.grid()
# plt.show()
fig.tight_layout()
fig.savefig("scaling_plot_logscale.pdf")
# fig.savefig("scaling_plot.pdf")
