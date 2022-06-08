import os
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from utils.exp_plot import (
    set_general_plot_style,
    incumbent_plot,
    save_fig
)

BASE_PATH = Path(__file__).parent.parent / "results"

EXPERIMENTS = {
    "RQ_1": ["RS", "RS_just_hpo", "RS_just_nas"],
    "RQ_2a": ["SH_Epochs", "SH_Epochs_just_hpo", "SH_Epochs_just_nas"],
    "RQ_2b": ["RS", "RS_just_hpo", "RS_just_nas",
              "SH_Epochs", "SH_Epochs_just_hpo", "SH_Epochs_just_nas"],
    "RQ_2": ["SH_Epochs", "SH_N", "SH_W", "SH_Resolution"]
}

# average configuration training runtime for each dataset
MAX_BUDGET = {
    "cifar10": 175571,
    "colorectal_histology": 18336,
    "fashionMNIST": 193248
}

Y_MAP = {
    "cifar10": {"RQ_1": [9, 20], "RQ_2": [9, 14], "RQ_2a": [9, 14], "RQ_2b": [9, 14]},
    "colorectal_histology": {"RQ_1": [4, 15], "RQ_2": [4, 10], "RQ_2a": [4, 10], "RQ_2b": [4, 10]},
    "fashionMNIST": {"RQ_1": [4, 15], "RQ_2": [4.75, 6], "RQ_2a": [4.5, 10], "RQ_2b": [4.5, 10]},
}

WIDTH_PT = 398.33864


def get_seed_info(path, seed, get_loss_from_run_fn=lambda r: r.loss):
    # load runs from log file
    result = hpres.logged_results_to_HBS_result(os.path.join(path, seed))
    # get all executed runs
    all_runs = result.get_all_runs()

    dataset = list(filter(
        None,
        list(map(lambda _d: _d if _d in path else None, MAX_BUDGET.keys()))
    ))[0]

    data = []
    for r in all_runs:
        if r.loss is None:
            continue
        c = r.info["cost"] / MAX_BUDGET[dataset]
        l = get_loss_from_run_fn(r)

        _id = r.config_id
        data.append((_id, c, l))

    if "Epochs" in path:
        data.reverse()
        for idx, (_id, c, l) in enumerate(data):
            for _i, _c, _ in data[data.index((_id, c, l)) + 1:]:
                if _i != _id:
                    continue
                data[idx] = (_id, c - _c, l)
                break
        data.reverse()

    data = [(d[1], d[2]) for d in data]
    cost, incumbent = zip(*data)

    return list(cost), list(np.minimum.accumulate(incumbent))


for experiment, strategies_to_plot in EXPERIMENTS.items():

    set_general_plot_style()

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(MAX_BUDGET.keys()),
        figsize=(5.3, 2.2),

    )

    for dataset_idx, dataset in enumerate(sorted(os.listdir(BASE_PATH))):
        if not os.path.isdir(os.path.join(BASE_PATH, dataset)):
            continue
        _base_path = os.path.join(BASE_PATH, dataset)
        for strategy_idx, strategy in enumerate(sorted(os.listdir(_base_path))):
            incumbents = []
            costs = []
            if strategy not in strategies_to_plot:
                continue
            if not os.path.isdir(os.path.join(_base_path, strategy)):
                continue
            _path = os.path.join(_base_path, strategy)

            for seed in sorted(os.listdir(_path)):
                cost, incumbent = get_seed_info(_path, seed)
                incumbents.append(incumbent)
                costs.append(cost)

            incumbent_plot(
                ax=axs[dataset_idx],
                x=np.array(costs),
                y=np.array(incumbents),
                title=dataset,
                xlabel="Runtime (full func evals)",
                ylabel="Val error [%]" if dataset_idx == 0 else None,
                strategy=strategy,
                log=False,
            )

            axs[dataset_idx].set_xlim(0, 100)
            axs[dataset_idx].set_ylim(
                min(Y_MAP[dataset][experiment]),
                max(Y_MAP[dataset][experiment])
            )

    sns.despine(fig)

    _legend_flag = len(strategies_to_plot) % 2 != 0
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15) if _legend_flag else (0.5, -0.25),
        ncol=len(strategies_to_plot) if _legend_flag else 2,
        frameon=False
    )
    fig.tight_layout(pad=0, h_pad=.5)
    save_fig(
        fig,
        filename=f"{experiment}",
        output_dir=os.path.join(BASE_PATH, "..", "plots")
    )
