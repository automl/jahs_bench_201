import json
from pathlib import Path
import argparse
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter, FormatStrFormatter
from matplotlib.lines import Line2D

import numpy as np

parser = argparse.ArgumentParser("")
parser.add_argument("--base", type=Path, help="The path to a directory which contains multiple sub-directories, "
                                              "each corresponding to a summarized data file called stats.json.")
args = parser.parse_args()
base: Path = args.base

# write_csv = True
write_csv = False

ypad = 1

image_sizes = ["img_size_8", "img_size_16", "img_size_32"]
channel_size_groups = ["cell_width_3", "cell_width_2", "cell_width_1"]
cell_repeats = ["cell_repeat_1", "cell_repeat_2", "cell_repeat_5"]
# cell_repeats = ["cell_repeat_1", "cell_repeat_2"]
# cell_repeats = ["cell_repeat_1"]
# processors = ["cpu", "cpu_2", "cpu_4", "gpu"]
processors = ["gpu",]
base_proc = "cpu"

# list(3): image_sizes -> list(3): channel_size_groups -> list(3): cell_repeats: float
data = []
for img_size in image_sizes:
    img_data = []
    for channel_size in channel_size_groups:
        times = []
        procs = []
        for i, sub in enumerate(cell_repeats, 1):
            filename = base / img_size / channel_size / sub / "stats.json"
            with open(filename, "r") as fp:
                sdata = json.load(fp)

            base_time = sdata[base_proc]["avg_train_time"]
            for proc in processors:
                avg_data = sdata[proc]
                times.append(base_time / avg_data["avg_train_time"])
                procs.append(proc)
        img_data.append((times, procs))
    data.append(img_data)

plt.rcdefaults()
plt.rcParams.update({"xtick.labelsize": 18, "ytick.labelsize": 18, "axes.labelsize": 20, "axes.titlesize": 20,
                     "legend.fontsize": 16})
nrows = 1
ncols = len(image_sizes)
fig, axs = plt.subplots(nrows, ncols, sharey=True, figsize=(16, 9))

labels_to_colors = {}
colors = iter(plt.cm.Set2.colors)

def map_label_to_color(label: str):
    if not label in labels_to_colors:
        labels_to_colors[label] = next(colors)
    return labels_to_colors[label]

image_size_label_map = {
    "img_size_8": "Image Size: 8x8",
    "img_size_16": "Image Size: 16x16",
    "img_size_32": "Image Size: 32x32",
}

channel_size_label_map = {
    "cell_width_1": "Channels: [16, 32, 64]",
    "cell_width_2": "Channels: [8, 16, 32]",
    "cell_width_3": "Channels: [8, 8, 16]",
}
group_size = len(processors)
xlabels = [int(s.replace("cell_repeat_", " ")) for s in cell_repeats]

for c, col_data in enumerate(data):
    for l, ((times, procs), channels) in enumerate(zip(col_data, channel_size_groups)):
        ax = axs[c]
        ax.plot(xlabels.copy(), times, label=channel_size_label_map[channels], linewidth=3, marker='*', markersize=14, linestyle="--")
        ax.set_yscale("log")

        # Decorate X-Axis
        ax.set_xticks(xlabels)
        ax.xaxis.grid(True, which="major", linewidth="0.5", color="red", linestyle="--")
        ax.yaxis.grid(True, which="major", linewidth="1.5")
        ax.yaxis.grid(True, which="minor", linestyle="--")
        ax.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        # Decorate shared Y-axis
        if c == 0:
            ax.set_ylabel("Speedup in training time, GPU / CPU")
            ax.legend()

        # Decorate center X-axis
        if c == ncols // 2:
            ax.set_xlabel("Network depth (# of cell repetitions - N)")

        ax.set_title(image_size_label_map[image_sizes[c]])

# Create Legend

fig.tight_layout()
fig.savefig(base / "comparison_frank.pdf")
# plt.show()

# Output CSV data
if write_csv:
    import csv
    gpu_idx = processors.index("gpu")
    num_processors = len(processors)
    num_cpu_types = num_processors - 1
    header1 = ["Channels", "# Cell Repetitions"]
    header2 = [" "]
    for label in y_major_labels:
        header2 += [label] + [" "] * (num_cpu_types - 1)

    tmp = processors.copy()
    tmp.pop(gpu_idx)
    header3 = [" "] + tmp * len(cell_repeats)
    rows = [header1, header2, header3]
    channel_size_label_map = {
        "cell_width_1": "16-32-64",
        "cell_width_2": "8-16-32",
        "cell_width_3": "8-8-16",
    }
    for group_idx, ((ypos, times, procs), channels) in enumerate(zip(data, channel_size_groups)):
        row = [channel_size_label_map[channels]]
        for subgroup_idx in range(0, len(ypos), num_processors):
            cpu_times = times[subgroup_idx:subgroup_idx + num_processors]
            gpu_time = cpu_times.pop(gpu_idx)
            row += [ctime / gpu_time for ctime in cpu_times]
        rows.append(row)

    with open(base / "summary.csv", "w") as csv_fp:
        csvwriter = csv.writer(csv_fp, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerows(rows)