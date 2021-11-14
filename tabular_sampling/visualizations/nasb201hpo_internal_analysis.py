"""
Expected structure of a single job's metric DataFrame:

index - pandas.MultiIndex, levels: [taskid, model_idx, Epoch]
columns - pandas.MultiIndex, level names: [MetricType, MetricName]
    where MetricType is generally [train, valid, test, diagnostic] and MetricName corresponds to values from
    tabular_sampling.lib.constants

"""

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from tabular_sampling.lib import constants
from tabular_sampling.lib.constants import MetricDFIndexLevels, metricdf_column_level_names, metricdf_index_levels

model_ids_by = [MetricDFIndexLevels.taskid.value, MetricDFIndexLevels.modelid.value]
fidelity_params = ("N", "W", "Resolution")
# fidelity_params=("N", "W")

# Common settings
plt.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16})

def map_label_to_color(label: str, map: Dict, colors: Iterable):
    if not label in map:
        map[label] = next(colors)
    return map[label]


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=Path, help="Path to the base directory where the relevant dataframe(s) "
                                                     "is/are stored as a file named data.pkl.gz.")
    args = parser.parse_args()
    return args


def analyze_single_job(basedir, save_tables=False, save_plots=False):
    """
    Analyzes a single job's output. A single job consists of any number of parallel, i.i.d. evaluations distributed
    across any number of nodes on the cluster on a joint HPO+NAS space. The data is expected to be read from a single
    DataFrame stored in "[basedir]/data.pkl.gz".
    """

    outdir = basedir / "analysis"
    outdir.mkdir(exist_ok=True, parents=True)
    df_fn = basedir / "data.pkl.gz"
    df: pd.DataFrame = pd.read_pickle(df_fn)
    try:
        assert isinstance(df.index, pd.MultiIndex), "DataFrame index must be a MultiIndex."
    except AssertionError as e:
        raise RuntimeError(f"Could not properly parse dataframe stored at '{df_fn}'") from e

    nepochs = df.groupby(model_ids_by)[df.columns.values[0]].agg("count").to_frame(("nepochs"))
    confs = df["model_config"].xs(1, level=MetricDFIndexLevels.epoch.value)
    valid_acc = df[("valid", "acc")].groupby(model_ids_by).agg("max").to_frame("valid-acc")
    test_acc = df[("test", "acc")].groupby(model_ids_by).agg("max").to_frame("test-acc")

    print(f"Valid accuracy stats: {valid_acc.describe()}")
    print(f"Test accuracy stats: {test_acc.describe()}")

    return nepochs, confs, valid_acc, test_acc
    #
    #
    # if "idx" in df.index.names:
    #     df_no_idx = df.xs(0, level="idx")
    # else:
    #     df_no_idx = df
    #
    # fidelity_group = df_no_idx.groupby(level=fidelity_params)
    # counts = fidelity_group.size()
    # if save_tables:
    #     table = counts.unstack(level=fidelity_params[-1]).to_latex(caption="Fidelity-wise sample count.")
    #     with open(outdir / "fidelity_count_table.tex", "w") as fp:
    #         fp.write(table)
    # else:
    #     print(f"Fidelity-wise sample count:\n{counts.unstack(level=fidelity_params[-1])}")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(16, 9)) if save_plots else plt.subplots(1, 1)
    # fig: plt.Figure
    # ax: plt.Axes
    #
    # bar_width = 1
    # bin_loc_offset = 1
    # xticks = [[], []]
    # labels_to_colors = {}
    # all_colors = iter(plt.cm.Set2.colors)
    #
    # for level_0_key in counts.index.unique(0):
    #     subseries = counts.xs(level_0_key, level=0)
    #     ser_colors = [map_label_to_color(i, labels_to_colors, all_colors) for i in subseries.keys()]
    #     bin_locs = list(range(bin_loc_offset, bin_loc_offset + subseries.size, bar_width))
    #     ax.bar(bin_locs, height=subseries.values, width=bar_width, color=ser_colors)
    #     bin_loc_offset += subseries.size + 2
    #     xticks[0] += [sum(bin_locs) / subseries.size, ]
    #     xticks[1] += [level_0_key, ]
    #
    # ax.xaxis.set_ticks(xticks[0])
    # ax.xaxis.set_ticklabels(xticks[1])
    #
    # legend_labels = [f"{counts.index.names[1]}={k}" for k in labels_to_colors.keys()]
    # legend_lines = [Line2D([0], [0], linewidth=8, color=labels_to_colors[l]) for l in labels_to_colors.keys()]
    # ax.legend(legend_lines, legend_labels, loc="upper right", fontsize=16)
    # ax.set_title("Number of samples drawn from each fidelity level.")
    # ax.set_xlabel("Number of cell-repetitions per architecture (N).")
    # ax.set_ylabel("Number of samples drawn from the search space.")
    # ax.yaxis.set_tick_params(which='minor')
    # # ax.yaxis.grid(True, which='major')
    # # ax.yaxis.grid(True, which='minor', linestyle="--", linewidth=2., color='gray')
    # if save_plots:
    #     fig.tight_layout()
    #     fig.savefig(outdir / "per_fidelity_sample_counts.pdf")
    # else:
    #     plt.show()
    #
    # ### Analyze average model training time ###
    # mean_train_times = df["train_time"].groupby(level=df.index.names.difference(["idx"])).mean()
    # mean_train_times: pd.Series = mean_train_times.groupby(level=fidelity_params).mean()
    #
    # if save_tables:
    #     table = mean_train_times.to_frame().unstack(fidelity_params[-1]).to_latex(
    #         caption="Average training time per epoch across fidelity values.")
    #     with open(outdir / "per_fidelity_training_time.tex") as fp:
    #         fp.write(table)
    # else:
    #     print(f"Fidelity-wise average model training time per epoch:\n"
    #           f"{mean_train_times.to_frame().unstack(fidelity_params[-1])}")

    # fig, ax = plt.subplots(1, 1, figsize=(16, 9)) if save_plots else plt.subplots(1, 1)
    # fig: plt.Figure
    # ax: plt.Axes


if __name__ == "__main__":
    args = parse_cli()
    basedir: Path = args.basedir
    analyze_single_job(basedir=basedir)
