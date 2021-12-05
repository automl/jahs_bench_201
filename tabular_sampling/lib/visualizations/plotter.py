"""
Contains the all visualizations for dataframes supported by the package. Currently available visualizations are:

'mean-std': A comparison of the mean and 1-sigma values of observed datapoints across the given indices. Accepts upto
three index names, using the first index to generate in-plot labels across curves, the second index to generate subplot
columns, and the third label to generate subplot rows, as needed. Technically applicable to any dataframe that contains
an index level named "iteration" if the other comparison indices are specified.

'hist-groups': Generates grouped histograms to compare data across up to three different parameters.
"""

import logging
from typing import List, Sequence, Any, Tuple, Union, Iterator, Dict, Optional, Hashable

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

plt.rcParams.update({
    "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 16, "axes.titlesize": 16, "axes.linewidth": 2,
    "axes.edgecolor": "black", "lines.linewidth": 5
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
    occupy an axes of its own. If position is "auto" (default), the position of the legend on the figure is determined
    automatically depending on the number of rows and columns in the grid. Otherwise, the values "bottom" or "right"
    can be specified. This must be set to None if no legend is required, in which case the value of legend_size is
    ignored. """

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
    elif pos == None:
        # No separate legend axis is required
        def adjust_gridspec(height, width, height_ratios, width_ratios) -> \
                Tuple[float, float, Sequence[float], Sequence[float]]:
            return nrows, ncols, height, width, height_ratios, width_ratios

        legend_kwargs = None
        legend_ax_loc = None
    else:
        raise RuntimeError("Unrecognized legend position %s" % pos)

    draw_nrows, draw_ncols, draw_area_height, draw_area_width, height_ratios, width_ratios = adjust_gridspec(
        cell_height * nrows, cell_width * ncols, [cell_height] * nrows, [cell_width] * ncols)

    # Padding for the various grid objects.
    wspace = 0.02
    hspace = 0.02
    legend_padding = legend_fontsize * pt_to_inch * 1.4

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


def draw_hist_groups(ax: plt.Axes, data: [pd.Series, pd.DataFrame], compare_indices: Sequence[str],
                     color_generator: Iterator, known_colors: dict, calculate_stats: bool = True,
                     label_bars: bool = True):
    """ Draw grouped histograms according to the given data. 'data' can be either a pandas Series or DataFrame object.
    In the case of the latter, it can contain up to two columns, 'mean' and 'std'. If only one column is present or if
    'data' is a Series, it is treated as the 'mean' column. 'compare_indices' contains up to 2 strings, corresponding
    to the index levels along which the histograms are grouped, such that each unique value of 'compare_indices[0]'
    corresponds to one group of histograms. 'color_generator' is an iterator that iteratres over matplotlib compatible
    colours and the dictionary 'known_colors' is used to check if a new colour should be generated. 'calculate_stats'
    is still unused and has been provided mostly for the purpose of compatibility with other functions. 'label_bars'
    determines whether or not the mean value represented by each bar should be labeled next to it. """

    assert isinstance(data, (pd.Series, pd.DataFrame)), \
        f"draw_hist_groups() expects the data to be passed as either a pandas Series or DataFrame, not " \
        f"{str(type(data))}"

    if isinstance(data, pd.Series):
        mean = data
        std = None
    else:
        data: pd.DataFrame
        assert len(data.columns) <= 2, "When data is passed as a DataFrame to draw_hist_groups(), it should contain " \
                                       "no more than two columns, optionally named 'mean' and 'std'."
        mean = data[data.columns[0]] if 'mean' not in data.columns else data['mean']
        std = pd.Series(None, index=mean.index) if data.columns.size == 1 else \
            data[data.columns[1]] if 'std' not in data.columns else data['std']

    draw_stds = not std is None

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
        submean: pd.Series = mean.xs(level_1_key, level=compare_indices[1]) if level_1_key is not None else mean
        substd: pd.Series = None if not draw_stds else std.xs(level_1_key, level=compare_indices[1]) \
            if level_1_key is not None else std

        level_0_labels = submean.index.get_level_values(compare_indices[0]).values
        ser_colors = [map_label_to_color(i, known_colors, color_generator) for i in level_0_labels]
        bin_locs = list(range(bin_loc_offset, bin_loc_offset + submean.size, bar_height))
        ax.barh(bin_locs, height=bar_height, width=submean.values, color=ser_colors, edgecolor="k", align="center",
                xerr=None if not draw_stds else substd, error_kw={"elinewidth": 1})
        if label_bars:
            # for x, y in zip(submean.values if not draw_stds else (submean + substd).values, bin_locs):
            for x, y in zip(submean.values, bin_locs):
                ax.text(x, y, f"{x:.2f}")
        bin_loc_offset += (submean.size + 2) * bar_height
        yticks[0] += [sum(bin_locs) / submean.size, ]
        yticks[1] += [level_1_key, ]

    ax.yaxis.set_ticks(yticks[0])
    ax.yaxis.set_ticklabels(yticks[1])
    ax.invert_yaxis()


def _mean_std_plot(ax: plt.Axes, data: pd.Series, across: str, xaxis_level: str, color_generator: Iterator,
                   known_colors: dict, calculate_stats: bool = True, outlines=True):
    """ Plots a Mean-Variance metric data visualization on the given Axes object comparing all indices defined by the
        name 'across' in the Series 'data' within the same plot. Remember that the Series index must contain at
        least 2 levels, one of which has to be 'across' and the other 'xaxis_level'. The remaining level(s) will be
        averaged over to generate the means. If 'calculate_stats' is True, the function calculates mean and std values
        on the fly. If it is False, the function expects the input data to have two columns, "mean" and "std"
        containing the respective values. """

    _log.info(f"Generating mean-std plot for {data.shape[0]} values, across the level {across}, using {xaxis_level} as "
              f"X-Axis.")

    labels = data.index.unique(level=across)
    if calculate_stats:
        groups = data.groupby([across, xaxis_level])
        mean_data: pd.Series = groups.mean()
        std_data: pd.Series = groups.std()
    else:
        mean_data = data["mean"]
        std_data = data["std"]

    for ctr, label in enumerate(labels):
        subset_means: pd.Series = mean_data.xs(label, level=across).sort_index(axis=0)
        subset_stds: pd.Series = std_data.xs(label, level=across).sort_index(axis=0)
        xs: np.ndarray = subset_means.index.to_numpy().squeeze()
        means: np.ndarray = subset_means.to_numpy().squeeze()
        std: np.ndarray = subset_stds.to_numpy().squeeze()
        colour = map_label_to_color(label, known_colors, color_generator)
        ax.plot(xs, means, c=colour, label=label, linewidth=linewidth)
        ax.fill_between(xs, means - std, means + std, alpha=0.2, color=colour)
        if outlines:
            ax.plot(xs, means - std, c=colour, alpha=0.6, linewidth=int(linewidth / 2))
            ax.plot(xs, means + std, c=colour, alpha=0.6, linewidth=int(linewidth / 2))
    if log_y:
        min_val, max_val = np.log10(mean_data.min() + 10 ** -6), np.log10(mean_data.max())
        range = max_val - min_val
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
        assert idx in index.names, f"{idx} is not a valid level name for the given dataframe with levels " \
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
                ax.set_title(f"{indices[1]}={clabel}", fontdict=dict(fontsize=label_fontsize))

            # Left-most column only
            # if cidx == 0:
            #     ax.set_ylabel(f"{indices[2]}={rlabel}", labelpad=10, fontdict=dict(fontsize=label_fontsize))

            # Right-most column only
            if cidx == ncols - 1 and nind == 3:
                alt_ax: plt.Axes = ax.twinx()
                alt_ax.set_ylabel(f"{indices[2]}={rlabel}", labelpad=10, fontdict=dict(fontsize=label_fontsize))
                alt_ax.yaxis.set_ticks([])

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
    fig.align_labels(axs=fig.axes[0::ncols])  # Align only the left-column's y-labels
    fig.set_constrained_layout_pads(w_pad=1.1 * label_fontsize * pt_to_inch, h_pad=1.1 * label_fontsize * pt_to_inch)
    fig.set_constrained_layout(True)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top', fontsize=label_fontsize + 10)

    return fig


class _AxisAligner:
    def __init__(self, which: str = 'x'):
        assert which in ('x', 'y', 'X', 'Y'), f"'which' must be either 'x' or 'y', was {which}."
        self.which = which.lower()
        self._known_axes = {}
        self._known_limits = {}

    def record(self, ax: plt.Axes, index: Hashable):
        self._known_axes[index] = ax
        self._known_limits[ax] = ax.get_xlim() if self.which == 'x' else ax.get_ylim()

    def align_limits(self, indices: Optional[Sequence[Hashable]] = None):
        if indices is None:
            return

        low, high = [], []
        for i in indices:
            ax = self._known_axes[i]
            low.append(self._known_limits[ax][0])
            high.append(self._known_limits[ax][1])

        lowest, highest = min(low), max(high)
        for i in indices:
            if self.which == 'x':
                self._known_axes[i].set_xlim(lowest, highest)
                # self._known_axes[i].xaxis.set_minor_locator(mtick.LinearLocator(10))
            else:
                self._known_axes[i].set_ylim(lowest, highest)
                # self._known_axes[i].yaxis.set_minor_locator(mtick.LinearLocator(10))


def hist_groups(data: Union[pd.Series, pd.DataFrame], indices: List[str], suptitle: str = None,
                legend_pos: str = "auto", xlabel: str = "", align_xlims: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Create a visualization that displays a grid of grouped histograms, such that each cell in the grid contains a
    single plot consisting of multiple groups of horizontal bars.

    :param data: pandas.DataFrame
        A pandas Series or DataFrame object containing all the data to be visualized with the appropriate index.
        Consult parameter 'indices' below and 'draw_hist_groups()' for more details.
    :param indices: A list of strings
        Upto four strings denoting the level names of a pandas Multi-Level Index across which comparisons are to be
        visualized. The first name is used to generate multiple bars in the same group, the second name for
        generating groups of bars, the third for comparisons across grid columns and the fourth for comparisons across
        rows.
    :param suptitle: string
        Used to attach a title for the visualization as a whole.
    :param legend_pos: str
        The position of the legend w.r.t. the grid. Possible values: "auto", "bottom", "right". Default: "auto".
    :param xlabel: str
        The common xlabel for all x-axes. All other axes labels are fixed to denote the various index level names and
        values.
    :param align_xlims: str or None
        If None (default), the limits of the x-axis (called xlims) are not adjusted and aligned across grid cells. If
        'row' or 'r', all xlims along each grid row are adjusted to be equal. Similarly and analogously for 'col' or
        'c', corresponding to columns. If 'both' is given, all xlims across all cells are aligned to the same values.

    :return: plt.Figure
        The Figure object containing the drawn histograms.
    """

    index: pd.MultiIndex = data.index
    labels_to_colors = kwargs.get("labels_to_colors", {})
    colors = kwargs.get("colors", iter(plt.cm.Set2.colors))

    # Identify the requested layout
    _log.info("Inferring plot layout.")

    nind = len(indices)
    assert nind <= 4, "Hist-Group visualization cannot handle more than 4 index names to compare across."
    if align_xlims not in [None, 'r', 'row', 'c', 'col', 'both']:
        raise ValueError(f"Unrecognized value for 'align_xlims': {align_xlims}")

    for idx in indices:
        assert idx in index.names, f"{idx} is not a valid level name for the given dataframe with levels {index.names}"

    col_labels, grid_col_idx_level = ([None], None) if nind < 3 else (index.unique(level=indices[2]), indices[2])
    row_labels, grid_row_idx_level = ([None], None) if nind < 4 else (index.unique(level=indices[3]), indices[3])
    grid_indices = [grid_row_idx_level, grid_col_idx_level]
    nrows = len(row_labels)
    ncols = len(col_labels)

    _log.info("Setting up plot.")
    legend_size = index.unique(level=indices[0]).size

    # Create the required Figure
    fig, gs, legend_ax_loc, legend_kwargs = create_figure(nrows=nrows, ncols=ncols, legend_size=legend_size,
                                                          pos=legend_pos)
    axis_aligner = _AxisAligner(which='x')

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
                                 known_colors=labels_to_colors)

            # Decorate X-Axis
            ax.xaxis.grid(True, which="major", linewidth="1.5")
            ax.xaxis.grid(True, which="minor", linestyle="--")

            # Bottom row only
            if ridx == nrows - 1:
                ax.set_xlabel(f"{xlabel}", labelpad=10., loc='right')
            else:
                ax.set_xlabel("")

            # Top row only
            if ridx == 0 and nind >= 3:
                ax.set_title(f"{indices[2]}={clabel}", fontdict=dict(fontsize=label_fontsize))

            # Left-most column only
            # TODO: Confirm if both ylabels are working properly
            if cidx == 0:
                ax.set_ylabel(indices[1], fontdict=dict(fontsize=label_fontsize))

            # Right-most column only
            if cidx == ncols - 1 and nind == 4:
                alt_ax: plt.Axes = ax.twinx()
                alt_ax.set_ylabel(f"{indices[3]}={rlabel}", labelpad=10, fontdict=dict(fontsize=label_fontsize))
                alt_ax.yaxis.set_ticks([])

            # Record X-axis limits
            if align_xlims is not None:
                axis_aligner.record(ax, (ridx, cidx))

    # Perform X-Axis limits alignment
    if align_xlims is not None:
        if align_xlims in ['r', 'row']:
            for r in range(nrows):
                axis_aligner.align_limits([(r, c) for c in range(ncols)])
        elif align_xlims in ['c', 'col']:
            for c in range(ncols):
                axis_aligner.align_limits([(r, c) for r in range(nrows)])
        elif align_xlims == 'both':
            axis_aligner.align_limits([(r, c) for r in range(nrows) for c in range(ncols)])

    # Draw legend
    legend_ax: plt.Axes = fig.add_subplot(gs[legend_ax_loc])
    legend_labels = [f"{indices[0]}={k}" for k in labels_to_colors.keys()]
    legend_lines = [plt.Line2D([0], [0], linewidth=8, color=labels_to_colors[l]) for l in labels_to_colors.keys()]
    _ = legend_ax.legend(legend_lines, legend_labels, **legend_kwargs)
    legend_ax.set_axis_off()
    fig.align_labels(axs=fig.axes[0::ncols])  # Align only the left-column's y-labels
    fig.set_constrained_layout_pads(w_pad=1.1 * label_fontsize * pt_to_inch, h_pad=1.1 * label_fontsize * pt_to_inch)
    fig.set_constrained_layout(True)
    if suptitle:
        fig.suptitle(suptitle, ha='center', va='top', fontsize=label_fontsize + 10)

    return fig


# TODO: Finish implementation
def time_series_stats_plot():
    fig, gs, legend_ax_loc, legend_kwargs = plotter.create_figure(2, 3, 6, pos="auto")
    legend = None
    for col, metric_type in enumerate(metric_type_map.keys()):
        if metric not in df[metric_type].columns:
            continue
        scale_data = metric_enable_scaling[metric]
        series_data: pd.Series = df[metric_type][metric]
        smoothed_series = series_data.groupby(lambda x: (x[0], (((x[1] - 1) // xaxis_smoothing_factor) + 1) *
                                                         xaxis_smoothing_factor)).agg("mean")
        smoothed_series.index = pd.MultiIndex.from_tuples(smoothed_series.index, names=["config_idx", xaxis_level])
        base = smoothed_series.where(
            smoothed_series.index.get_level_values(xaxis_level) == xaxis_smoothing_factor).ffill()
        if scale_data:
            scaled_series = smoothed_series / base
            epoch_stats = scaled_series.groupby(xaxis_level).describe()
        else:
            epoch_stats = series_data.groupby(xaxis_level).describe()

        ax: plt.Axes = fig.add_subplot(gs[0, col])
        ax.set_title(metric_type_map[metric_type], fontsize=plotter.label_fontsize + 5)

        plotter._mean_std_plot(ax=ax, data=epoch_stats.assign(id="Mean").set_index("id", append=True), across="id",
                               xaxis_level=xaxis_level, color_generator=iter(plt.cm.Set2.colors), known_colors={},
                               calculate_stats=False)
        ax.set_xlabel(xaxis_level, fontsize=plotter.label_fontsize)
        if col == 0:
            h, l = ax.get_legend_handles_labels()
            legend = h, l

        ax: plt.Axes = fig.add_subplot(gs[1, col])
        colors = iter(plt.cm.Set2.colors)
        ax.plot(epoch_stats.index, epoch_stats["min"], c=next(colors), linestyle="--", linewidth=plotter.linewidth,
                label="Min")
        ax.plot(epoch_stats.index, epoch_stats["25%"], c=next(colors), linestyle="--", linewidth=plotter.linewidth,
                label="q=0.25")
        ax.plot(epoch_stats.index, epoch_stats["50%"], c=next(colors), linestyle="--", linewidth=plotter.linewidth,
                label="q=0.50")
        ax.plot(epoch_stats.index, epoch_stats["75%"], c=next(colors), linestyle="--", linewidth=plotter.linewidth,
                label="q=0.75")
        ax.plot(epoch_stats.index, epoch_stats["max"], c=next(colors), linestyle="--", linewidth=plotter.linewidth,
                label="Max")
        # ax.set_yscale("log")
        ax.set_xlabel(xaxis_level, fontsize=plotter.label_fontsize)
        ax.grid(True, which='both', linewidth=0.5, c='k')
        if col == 0:
            ax.set_ylabel(f"{metric_map[metric]}{' (Scaled) ' if scale_data else ''}", fontsize=plotter.label_fontsize)
            h, l = ax.get_legend_handles_labels()
            legend = legend[0] + h, legend[1] + l
        # ax.legend(fontsize=plotter.legend_fontsize)
        # ax.set_ylim(0.5, 1.5)

    legend_ax: plt.Axes = fig.add_subplot(gs[legend_ax_loc])
    handles, labels = legend
    _ = legend_ax.legend(handles, labels, **legend_kwargs)
    legend_ax.set_axis_off()

    fig.suptitle(f"Scaling analysis - {metric_map[metric]} across {df.index.unique(xaxis_level).size} Epochs for "
                 f"{df.index.unique('config_idx').size} Configurations.", fontsize=plotter.label_fontsize + 10)
    fig.align_labels(fig.axes)


# TODO: Polish and standardize
def plot_spearman_rank_correlation(rho: pd.DataFrame, pval: pd.DataFrame,
                                   exclude_row_vals: Union[None, Sequence[Any]] = None,
                                   exclude_col_vals: Union[None, Sequence[Any]] = None):
    rows = rho.index
    columns = rho.columns

    if exclude_row_vals is not None:
        rows = rows.difference(exclude_row_vals)

    if exclude_col_vals is not None:
        columns = columns.difference(exclude_col_vals)

    # Ensure there are no mix-ups in order
    rho = rho.loc[rows, columns]
    pval = pval.loc[rows, columns]

    nrows = len(rows)
    ncols = len(columns)

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    fig: plt.Figure
    for ax, (data, bounds) in zip(axs, [(rho.values, (-1., 1.)), (pval.values, (0., 1.))]):
        ax: plt.Axes
        mesh = ax.pcolormesh(data, vmin=bounds[0], vmax=bounds[1])
        fig.colorbar(mesh, ax=ax)
        ax.set_xticks(np.arange(ncols) + 0.5)
        ax.set_xticklabels(columns, rotation=45 if columns.dtype in [pd.StringDtype, pd.CategoricalDtype] else 0)
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels(rows)
        assert np.isnan(data).any().any() == False, "Found NaNs"

        colorbar_midpoint = sum(bounds) / 2

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j + 0.5, i + 0.5, "%.4f" % data[i, j],
                        fontdict={"ha": "center", "va": "center", "fontsize": 12,
                                  "color": "black" if data[i, j] >= colorbar_midpoint else "white"})

    axs[0].set_title("Spearman's Rank Correlation Comparison.")
    axs[1].set_title("p-Values of the respective correlations.")
    return fig
