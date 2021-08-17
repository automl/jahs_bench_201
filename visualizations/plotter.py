"""
Contains the all visualizations for dataframes supported by the package. Currently available visualizations are:

'mean-std': A comparison of the mean and 1-sigma values of observed datapoints across the given indices. Accepts upto
three index names, using the first index to generate in-plot labels across curves, the second index to generate subplot
columns, and the third label to generate subplot rows, as needed. Technically applicable to any dataframe that contains
an index level named "iteration" if the other comparison indices are specified.

'hist-groups': Generates grouped histograms to compare data across up to three different parameters.
"""


import logging
from typing import List, Sequence, Any, Tuple, Union, Iterator, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

_log = logging.getLogger(__name__)

plt.rcParams.update({
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16, "axes.linewidth": 2,
    "axes.edgecolor":"black", "lines.linewidth": 5
})

default_metrics_row_index_labels: Sequence[str] = ("model", "metric", "rng_offset", "iteration")
linewidth = 4.
log_y = True

cell_height = 4
cell_width = 8
label_fontsize = 20
legend_fontsize = 15
pt_to_inch = 0.04167 / 3

def map_label_to_color(label: str, map: Dict, colors: Iterator):
    if not label in map:
        map[label] = next(colors)
    return map[label]


def create_figure(nrows: int, ncols: int, legend_size: int, pos: str = "auto") -> \
        Tuple[plt.Figure, plt.GridSpec, Tuple[int, slice], dict]:
    """ Creates a Figure and accompanying GridSpec that conforms to the desired geometry, as specified by 'nrows' and
    'ncols'. The position and size of the legend must also be determined in advance since the legend is expected to
    occupy an axes of its own. If position is "auto", the position of the legend on the figure is determined
    automatically depending on the number of rows and columns in the grid. Otherwise, the values "bottom" or "right"
    can be specified. """

    if pos == "auto":
        pos = "bottom" if ncols >= nrows else "right"

    if pos == "bottom":
        # Legend will be placed below the grid
        legend_ax_height = cell_height / 8 * (legend_size // ncols + int(legend_size % ncols > 0))
        def adjust_gridspec(height, width, height_ratios, width_ratios) -> \
                Tuple[int, int, float, float, Sequence[float], Sequence[float]]:
            return nrows + 1, ncols, height + legend_ax_height, width, height_ratios + [legend_ax_height], width_ratios

        legend_kwargs = dict(loc='lower center', ncol=ncols, fontsize=legend_fontsize)
        legend_ax_loc = -1, slice(None)

    elif pos == "right":
        # Legend will be placed to the right of the grid
        legend_ax_width = cell_width / 2
        def adjust_gridspec(height, width, height_ratios, width_ratios) -> \
                Tuple[float, float, Sequence[float], Sequence[float]]:
            return nrows, ncols + 1, height, width + legend_ax_width, height_ratios, width_ratios + [legend_ax_width]

        legend_kwargs = dict(loc='center', ncol=1, fontsize=legend_fontsize)
        legend_ax_loc = slice(None), -1
    else:
        raise RuntimeError("Unrecognized legend position %s" % pos)

    draw_nrows, draw_ncols, draw_area_height, draw_area_width, height_ratios, width_ratios = adjust_gridspec(
        cell_height * nrows, cell_width * ncols, [cell_height] * nrows, [cell_width] * ncols)

    # Padding for the various grid objects.
    wspace = 0.02 * draw_area_width
    hspace = 0.02 * draw_area_height
    legend_padding = legend_fontsize * pt_to_inch * 1.2

    plt.rcParams["figure.figsize"] = (draw_area_width + legend_padding, draw_area_height + legend_padding)
    fig: plt.Figure = plt.figure(constrained_layout=True, frameon=True)
    gs: plt.GridSpec = plt.GridSpec(nrows=draw_nrows, ncols=draw_ncols, figure=fig, wspace=wspace, hspace=hspace,
                                    height_ratios=height_ratios)

    return fig, gs, legend_ax_loc, legend_kwargs


def get_view_on_data(row_val, col_val, data: Union[pd.DataFrame, pd.Series], grid_indices: Tuple[str, str]) -> \
        pd.DataFrame:
    """ Returns a cross-section of the full dataframe index using the given values of the comparison indices. """

    if col_val is None:
        return data
    if row_val is None:
        selection = (col_val), grid_indices[1]
    else:
        selection = (row_val, col_val), (grid_indices[0], grid_indices[1])
    return data.xs(selection[0], level=selection[1])


def draw_hist_groups(ax: plt.Axes, data: pd.Series, compare_indices: Sequence[str], color_generator: Iterator,
                   known_colors: dict, calculate_stats: bool = True):

    assert isinstance(data, pd.Series), f"draw_hist_groups() expects the data to be passed as a pandas Series, not " \
                                        f"{str(type(data))}"

    if len(compare_indices) > 2:
        raise RuntimeError("Cannot draw histogram groups with more than two levels of indices to be compared.")

    if len(compare_indices) == 1:
        # Make the second level index None.
        compare_indices = compare_indices + [None]
        level_1_labels = [None]
    else:
        level_1_labels = data.index.unique(compare_indices[1])

    bar_height = 1
    bin_loc_offset = 1
    yticks = [[], []]

    for level_1_key in level_1_labels:
        subseries: pd.Series = data.xs(level_1_key, level=compare_indices[1]) if level_1_key is not None else data

        level_0_labels = subseries.index.get_level_values(compare_indices[0]).values
        ser_colors = [map_label_to_color(i, known_colors, color_generator) for i in level_0_labels]
        bin_locs = list(range(bin_loc_offset, bin_loc_offset + subseries.size, bar_height))
        ax.barh(bin_locs, height=bar_height, width=subseries.values, color=ser_colors, edgecolor="k", align="center")
        bin_loc_offset += subseries.size + 2
        yticks[0] += [sum(bin_locs) / subseries.size, ]
        yticks[1] += [level_1_key, ]

    ax.yaxis.set_ticks(yticks[0])
    ax.yaxis.set_ticklabels(yticks[1])
    ax.invert_yaxis()

    # group_size = len(processors)
    # for c, col_data in enumerate(data):
    #     for r, ((ypos, times, (stds, mins, maxs), types), channels) in enumerate(zip(col_data, channel_size_groups)):
    #         ax = axs[r][c]
    #         if graph_type == "mean-std":
    #             ax.barh(ypos, times, height=0.9, align="center", xerr=stds, color=list(map(map_label_to_color, types)),
    #                     edgecolor="k")
    #         elif graph_type == "min-max":
    #             widths = np.array(maxs) - np.array(mins)
    #             ax.barh(ypos, width=widths, left=mins, height=0.9, align="center",
    #                     color=list(map(map_label_to_color, types)), edgecolor="k")
    #         else:
    #             raise ValueError(f"Unknown graph type '{graph_type}'. Must be one of 'mean-std' or 'min-max'.")
    #
    #         # Decorate X-Axis
    #         ax.xaxis.set_minor_locator(MultipleLocator(100))

    #
    #         # Decorate shared Y-axis
    #         if c == 0:
    #             yticks = [sum(ypos[i:i + group_size]) / group_size for i in range(0, len(ypos), group_size)]
    #
    #             ax.set_yticks(yticks)
    #             y_major_labels = [s.replace("cell_repeat_", " ") for s in cell_repeats]
    #             ax.set_yticklabels(y_major_labels, minor=False, size=14, va="center")
    #
    #         if c == ncols - 1:
    #             alt_ax: plt.Axes = ax.twinx()
    #             alt_ax.set_ylabel(channel_size_label_map[channels])
    #             alt_ax.yaxis.set_ticks([])
    #
    #         ax.invert_yaxis()
    #
    #         if r == nrows // 2 and c == 0:
    #             ax.set_ylabel("# Cell repetitions")
    #
    #         # Decorate shared X-axis
    #         if r == nrows - 1 and c == ncols // 2:
    #             ax.set_xlabel("Average training time (seconds)")
    #
    #         if r == 0:
    #             ax.set_title(image_size_label_map[image_sizes[c]])


def _mean_std_plot(ax: plt.Axes, data: pd.Series, across: str, xaxis_level: str, color_generator: Iterator,
                   known_colors: dict, calculate_stats: bool = True):
    """ Plots a Mean-Variance metric data visualization on the given Axes object comparing all indices defined by the
        name 'across' in the Series 'data' within the same plot. Remember that the Series index must contain at
        least 2 levels, one of which has to be 'across' and the other 'xaxis_level'. The remaining level(s) will be
        averaged over to generate the means. If 'calculate_stats' is True, the function calculates mean and std values
        on the fly. If it is False, the function expects the input data to have two columns, "mean" and "std"
        containing the respective values. """

    _log.info(f"Generating mean-std plot for {data.shape[0]} values, across the level {across}, using {xaxis_level} as "
              f"X-Axis.")

    labels = data.index.unique(level=across)
    groups = data.groupby([across, xaxis_level])
    if calculate_stats:
        mean_data: pd.Series = groups.mean()
        std_data: pd.Series = groups.std()
    else:
        mean_data = groups.loc[:, "mean"]
        std_data = groups.loc[:, "std"]

    min_val, max_val = np.log10(mean_data.min() + 10 ** -6), np.log10(mean_data.max())
    range = max_val - min_val

    for ctr, label in enumerate(labels):
        subset_means: pd.Series = mean_data.xs(label, level=across).sort_index(axis=0)
        subset_stds: pd.Series = std_data.xs(label, level=across).sort_index(axis=0)
        xs: np.ndarray = subset_means.index.to_numpy().squeeze()
        means: np.ndarray = subset_means.to_numpy().squeeze()
        std: np.ndarray = subset_stds.to_numpy().squeeze()
        colour = map_label_to_color(label, known_colors, color_generator)
        ax.plot(xs, means, c=colour, label=label, linewidth=linewidth)
        ax.fill_between(xs, means - std, means + std, alpha=0.2, color=colour)
    if log_y:
        ax.set_yscale('log')
        mtick.LogFormatter()
        formatter = mtick.LogFormatterMathtext(labelOnlyBase=False, minor_thresholds=(2 * range, 0.5 * range))
        ax.yaxis.set_major_formatter(formatter)
        formatter.set_locs()
        ax.yaxis.set_major_locator(mtick.LogLocator(numticks=3))
        ax.yaxis.set_minor_locator(mtick.LogLocator(subs='all', numticks=10))
        current_y_lim = ax.get_ylim()[0]
        ylim = np.clip(current_y_lim, 10 ** (min_val - 1), None)
        ax.set_ylim(bottom=ylim)
    else:
        formatter = mtick.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))
        formatter.set_scientific(True)
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, which='both', linewidth=0.5, c='k')


def mean_std(data: pd.DataFrame, indices: List[str], xaxis_level: str, suptitle: str = None,
             calculate_stats: bool = True, legend_pos: str = "auto") -> plt.Figure:
    """
    Create a visualization that displays the mean and 1-std envelope of the given data, possibly comparing across up to
    three individual dimensions.
    :param data: pandas.DataFrame
        A DataFrame object containing all the data to be visualized with the appropriate index.
    :param indices: A list of strings
        Upto three strings denoting the names of a pandas Multi-Level Index across which comparisons are to be
        visualized. The first name is used to generate comparisons within the same plot, the second name for
        comparisons across columns and the third for comparisons across rows.
    :param suptitle: string
        Used to attach a title for the visualization as a whole.
    :param xaxis_level: string
        A string that specifies the level of the index which is used to obtain values along the x-axis of the plots.
    :param calculate_stats: bool
        If 'calculate_stats' is True, the function calculates mean and std values on the fly. If it is False, the
        function expects the input data to have two columns, "mean" and "std" containing the respective values.
    :param legend_pos: str
        The position of the legend w.r.t. the grid. Possible values: "auto", "bottom", "right". Default: "auto".
    :return: None
    """

    index: pd.MultiIndex = data.index
    labels_to_colors = {}
    colors = iter(plt.cm.Set2.colors)

    # Identify the requested layout
    _log.info("Inferring plot layout.")

    nind = len(indices)
    assert nind <= 3, "Mean-Variance visualization of metric values cannot handle more than 3 " \
                      "index names to compare across."
    for idx in indices:
        assert idx in index.names, f"{idx} is not a valid level name for the given dataframe with levels "\
                                   f"{index.names}"

    col_labels: Sequence[Any] = index.unique(level=indices[1]) if len(indices) > 1 else [None]
    row_labels: Sequence[Any] = index.unique(level=indices[2]) if len(indices) > 2 else [None]
    across_idx = indices[0]
    grid_indices = indices[:0:-1]
    nrows = len(row_labels)
    ncols = len(col_labels)

    _log.info("Setting up plot.")
    legend_size = index.unique(level=across_idx).size

    # Create the required Figure
    fig, gs, legend_ax_loc, legend_kwargs = create_figure(nrows=nrows, ncols=ncols, legend_size=legend_size,
                                                          pos=legend_pos)

    legend = None
    for ridx, rlabel in enumerate(row_labels):
        for cidx, clabel in enumerate(col_labels):
            ax: plt.Axes = fig.add_subplot(gs[ridx, cidx])
            try:
                view = get_view_on_data(row_val=rlabel, col_val=clabel, data=data, grid_indices=grid_indices)
            except KeyError as e:
                # No data found for the requested key. Silently ignore it.
                pass
            else:
                _mean_std_plot(ax=ax, data=view, across=across_idx, xaxis_level=xaxis_level, color_generator=colors,
                               known_colors=labels_to_colors, calculate_stats=calculate_stats)

            # Bottom row only
            if ridx == nrows - 1:
                ax.set_xlabel(xaxis_level, labelpad=10., loc='right')
            else:
                ax.set_xticklabels([])

            # Top row only
            if ridx == 0:
                ax.set_title(clabel, fontdict=dict(fontsize=label_fontsize))

            # Left-most column only
            if cidx == 0:
                ax.set_ylabel(rlabel, labelpad=10, fontdict=dict(fontsize=label_fontsize))

            # This ensures that we don't miss any labels because different subplots had different subsets of labels.
            h, l = ax.get_legend_handles_labels()
            if legend is None:
                legend = h, l
            elif len(h) > len(legend[0]):
                legend = h, l

    legend_ax: plt.Axes = fig.add_subplot(gs[legend_ax_loc])
    handles, labels = legend
    _ = legend_ax.legend(handles, labels, **legend_kwargs)
    legend_ax.set_axis_off()
    fig.align_labels(axs=fig.axes[0::ncols]) # Align only the left-column's y-labels
    fig.set_constrained_layout_pads(w_pad=1.1 * label_fontsize * pt_to_inch, h_pad=1.1 * label_fontsize * pt_to_inch)
    fig.set_constrained_layout(True)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')

    return fig


def hist_groups(data: pd.DataFrame, indices: List[str], suptitle: str = None,
             calculate_stats: bool = True, legend_pos: str = "auto", xlabel: str = "") -> plt.Figure:
    """
    Create a visualization that displays the mean and 1-std envelope of the given data, possibly comparing across up to
    three individual dimensions.
    :param data: pandas.DataFrame
        A DataFrame object containing all the data to be visualized with the appropriate index.
    :param indices: A list of strings
        Upto three strings denoting the names of a pandas Multi-Level Index across which comparisons are to be
        visualized. The first name is used to generate comparisons within the same plot, the second name for
        comparisons across columns and the third for comparisons across rows.
    :param suptitle: string
        Used to attach a title for the visualization as a whole.
    :param xaxis_level: string
        A string that specifies the level of the index which is used to obtain values along the x-axis of the plots.
    :param calculate_stats: bool
        If 'calculate_stats' is True, the function calculates mean and std values on the fly. If it is False, the
        function expects the input data to have two columns, "mean" and "std" containing the respective values.
    :param legend_pos: str
        The position of the legend w.r.t. the grid. Possible values: "auto", "bottom", "right". Default: "auto".
    :return: None
    """

    index: pd.MultiIndex = data.index
    labels_to_colors = {}
    colors = iter(plt.cm.Set2.colors)

    # Identify the requested layout
    _log.info("Inferring plot layout.")

    nind = len(indices)
    assert nind <= 4, "Hist-Group visualization cannot handle more than 4 index names to compare across."
    for idx in indices:
        assert idx in index.names, f"{idx} is not a valid level name for the given dataframe with levels {index.names}"

    col_labels: Sequence[Any] = index.unique(level=indices[2]) if len(indices) > 2 else [None]
    row_labels: Sequence[Any] = index.unique(level=indices[3]) if len(indices) > 3 else [None]
    grid_indices = [indices[3], indices[2]]
    nrows = len(row_labels)
    ncols = len(col_labels)

    _log.info("Setting up plot.")
    legend_size = index.unique(level=indices[0]).size

    # Create the required Figure
    fig, gs, legend_ax_loc, legend_kwargs = create_figure(nrows=nrows, ncols=ncols, legend_size=legend_size,
                                                          pos=legend_pos)

    legend = None
    for ridx, rlabel in enumerate(row_labels):
        for cidx, clabel in enumerate(col_labels):
            ax: plt.Axes = fig.add_subplot(gs[ridx, cidx])
            try:
                view = get_view_on_data(row_val=rlabel, col_val=clabel, data=data, grid_indices=grid_indices)
            except KeyError as e:
                # No data found for the requested key. Silently ignore it.
                pass
            else:
                draw_hist_groups(ax=ax, data=view, compare_indices=indices[:2], color_generator=colors,
                               known_colors=labels_to_colors, calculate_stats=calculate_stats)

            # Decorate X-Axis
            ax.xaxis.grid(True, which="major", linewidth="1.5")
            ax.xaxis.grid(True, which="minor", linestyle="--")

            # Bottom row only
            if ridx == nrows - 1:
                ax.set_xlabel(f"{indices[1]}, {xlabel}", labelpad=10., loc='left')
            else:
                ax.set_xlabel("")

            # Top row only
            if ridx == 0:
                ax.set_title(f"{indices[2]}={clabel}", fontdict=dict(fontsize=label_fontsize))

            # Left-most column only
            if cidx == 0:
                ax.set_ylabel(f"{indices[3]}={rlabel}", labelpad=10, fontdict=dict(fontsize=label_fontsize))

    legend_ax: plt.Axes = fig.add_subplot(gs[legend_ax_loc])
    legend_labels = [f"{indices[0]}={k}" for k in labels_to_colors.keys()]
    legend_lines = [plt.Line2D([0], [0], linewidth=8, color=labels_to_colors[l]) for l in labels_to_colors.keys()]
    _ = legend_ax.legend(legend_lines, legend_labels, **legend_kwargs)
    legend_ax.set_axis_off()
    fig.align_labels(axs=fig.axes[0::ncols]) # Align only the left-column's y-labels
    fig.set_constrained_layout_pads(w_pad=1.1 * label_fontsize * pt_to_inch, h_pad=1.1 * label_fontsize * pt_to_inch)
    fig.set_constrained_layout(True)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top')

    return fig
