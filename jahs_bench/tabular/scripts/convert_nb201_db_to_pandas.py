# Note: This script requires an installation of the complete NASLib repository in order
# to work.
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.search_spaces.nasbench201 import conversions
from jahs_bench.tabular.lib.naslib.utils.utils import get_project_root
from pathlib import Path
import pickle
import pandas as pd

ss = NasBench201SearchSpace()

nb201_to_ops = {
        'avg_pool_3x3': 'AvgPool1x1',
        'nor_conv_1x1': 'ReLUConvBN1x1',
        'nor_conv_3x3': 'ReLUConvBN3x3',
        'skip_connect': 'Identity',
        'none': 'Zero',
    }

def recover_ops(model_str: str):
    ops = [s for s in model_str.split("|") if s not in ["", "+"]]
    split_ops = {i: [] for i in range(3)}
    for op, i in [s.split("~") for s in ops]:
        split_ops[int(i)].append(conversions.OP_NAMES.index(nb201_to_ops[op]))

    final_ops = []
    for i in range(3):
        final_ops += split_ops[i]

    return tuple(final_ops)

pth = Path(get_project_root()) / "data"
with open(pth / "nb201_all.pickle", "rb") as fp:
    rawdata = pickle.load(fp)

df = pd.DataFrame(rawdata).transpose()
df.index = pd.MultiIndex.from_tuples([recover_ops(k) for k, v in df.iterrows()])
dfs = []
for dataset in df.columns:
    dfs.append(
        pd.DataFrame.from_records(df[dataset].values, index=df.index).assign(dataset=dataset).set_index("dataset",
                                                                                                        append=True))

df = pd.concat(dfs)
df.to_pickle(Path("/home/archit/thesis") / "nasb201_full_pandas.pkl.gz")
