import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from path import Path
from scipy import stats

from .styles import STRATEGIES, COLOR_MARKER_DICT, DATASETS


def set_general_plot_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("deep")
    plt.switch_backend("pgf")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            "font.serif": [],
            "font.sans-serif": [],
            "font.monospace": [],
            "font.size": "10.90",
            "legend.fontsize": "9.90",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.title_fontsize": "small",
            # "bottomlabel.weight": "normal",
            # "toplabel.weight": "normal",
            # "leftlabel.weight": "normal",
            # "tick.labelweight": "normal",
            # "title.weight": "normal",
            "pgf.preamble": r"""
                \usepackage[T1]{fontenc}
                \usepackage[utf8x]{inputenc}
                \usepackage{microtype}
            """,
        }
    )


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.
    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


SAVE_DIR = '.'  # TODO:
def save_fig(fig, filename, output_dir=SAVE_DIR, dpi: int = 100):
    output_dir = Path(output_dir)
    output_dir.makedirs_p()
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", dpi=dpi)
    print(f'Saved to "{output_dir}/{filename}.pdf"')


def interpolate_time(incumbents, costs):
    df_dict = {}

    for i, _ in enumerate(incumbents):
        _seed_info = pd.Series(incumbents[i], index=np.cumsum(costs[i]))
        df_dict[f"seed{i}"] = _seed_info
    df = pd.DataFrame.from_dict(df_dict)

    df = df.fillna(method="backfill", axis=0).fillna(method="ffill", axis=0)
    return df


def incumbent_plot(
    ax,
    x,
    y,
    xlabel=None,
    ylabel=None,
    title=None,
    strategy=None,
    log=False,
    **plot_kwargs,
):
    df = interpolate_time(incumbents=y, costs=x)
    y_mean = df.mean(axis=1)
    std_error = stats.sem(df.values, axis=1)
    ax.plot(df.index, y_mean, label=STRATEGIES[strategy], **plot_kwargs, color=COLOR_MARKER_DICT[strategy]
            )
    ax.fill_between(
        df.index,
        y_mean - std_error,
        y_mean + std_error,
        color=COLOR_MARKER_DICT[strategy],
        alpha=0.2,
    )

    if title is not None:
        ax.set_title(DATASETS[title])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="-", alpha=0.8)

    if log:
        # ax.set_xscale("log")
        ax.set_yscale("log")
