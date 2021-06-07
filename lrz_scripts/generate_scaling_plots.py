import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

pth = Path("/home/archit/thesis/experiments/scaling_plots_stats")
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

x = np.array(nnodes)
order = x.argsort()
x = x[order]
y = np.array(nconfigs)[order]

# Dark Mode
plt.rcParams.update({"axes.facecolor": "black", "figure.facecolor": "black", "axes.labelcolor": "white",
                     "axes.titlecolor": "white", "axes.edgecolor": "silver", "text.color": "white",
                     "xtick.color": "white", "ytick.color": "white", "grid.color": "silver",
                     "xtick.labelsize": 12, "ytick.labelsize": 12})
fig, ax = plt.subplots(1, 1)
ax: plt.Axes

ax.plot(x, y, marker="*", markerfacecolor="red", markeredgecolor="red", linewidth=2, markersize=12, linestyle="dashed", color="blue")
ax.set_xlabel("Number of nodes", size=14)
ax.set_ylabel("Throughput", size=14)
ax.set_title("Scaling plot for throughput over all fidelity levels.", size=16)
# ax.set_yscale()
ax.grid()
# plt.show()
fig.tight_layout()
fig.savefig("scaling_plot.pdf")
