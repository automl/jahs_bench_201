import pandas as pd
import json
from pathlib import Path
import argparse
from typing import Union

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=Path, help="Path to the base directory where all the tasks' outputs are stored. "
                                                 "Complete DFS traversal should yield JSON files.")
args = parser.parse_args()

basedir: Path = args.basedir

# JSON file structure
valid_data_keys = sorted(["train_acc", "train_loss", "val_acc", "val_loss", "test_acc", "test_loss", "runtime",
                          "train_time", "model_size_MB", "config"])
valid_meta_keys = sorted(["epochs", "resize", "init_duration", "wc_duration", "proc_duration"])

def parse_output_file(file: Path, raise_warning=False) -> Union[None, dict, pd.DataFrame]:
    """
    Given a Path-like object for a file, attempts to parse the file as an output JSON file of the expected format and
    return the results as a pandas DataFrame. If parsing fails and raise_warning is False (default), returns None. If
    raise_warning is True, a warning is raised indicating why parsing failed and None is returned. In case a
    configuration could not be evaluated and thus registered an error in the output data, the JSON file is interpreted
    as a dict and returned instead.
    """

    try:
        with open(file) as fp:
            json_data: dict = json.load(fp)
            keys = sorted(json_data.keys())
        assert isinstance(json_data, dict), f"Output data should be stored as a dict in JSON files, " \
                                            f"found {type(json_data)}."
        if "exception" in keys:
            # In the case of an aborted configuration, returning a DataFrame makes no sense.
            return json_data
        elif not valid_data_keys == keys:
            # Only run the check for which keys are missing when we know that some keys are missing when they
            # shouldn't be.
            # Just to be sure, use a slightly different check for the keys the second time around.
            for k in valid_data_keys:
                assert k in json_data, f"Mandatory data key '{k}' not found in JSON file."
    except json.decoder.JSONDecodeError as e:
        if raise_warning:
            raise RuntimeWarning(f"File {str(file)} is not a valid JSON file.") from e
        return None
    except AssertionError as e:
        if raise_warning:
            raise RuntimeWarning(f"JSON file {str(file)} does not contain valid output data.") from e
        return None

    # DataFrame structure
    df_data = {}
    config = None
    for key, val in json_data.items():
        if key == "config":
            assert config is None, f"Cannot process a second config {str(val)} into an existing row index for the " \
                                    f"DataFrame - {str(row_idx)}"
            # row_idx = pd.MultiIndex.from_tuples([tuple(val.values())], names=list(val.keys()))
            config = val
        elif isinstance(val, list):
            df_data[key] = pd.Series(data=val)
        else:
            df_data[key] = val

    df = pd.DataFrame.from_dict(df_data)
    df.index.set_names("idx", inplace=True)
    df = df.assign(**config).set_index(keys=list(config.keys()), append=True)

    # row_idx = pd.MultiIndex.from_product([row_idx.values, df.index.values], names=[row_idx.names, "idx"])
    # df = df.set_index(row_idx)
    return df


def parse_meta_file(file: Path, raise_warning=False) -> Union[None, dict]:

    try:
        with open(file) as fp:
            json_data: dict = json.load(fp)
            keys = sorted(json_data.keys())
        assert isinstance(json_data, dict), f"Metadata should be stored as a dict in JSON files, " \
                                            f"found {type(json_data)}."
        if not valid_meta_keys == keys:
            # Only run the check for which keys are missing when we know that some keys are missing when they
            # shouldn't be.
            # Just to be sure, use a slightly different check for the keys the second time around.
            for k in valid_meta_keys:
                assert k in json_data, f"Mandatory metadata key '{k}' not found in JSON file."
    except json.decoder.JSONDecodeError as e:
        if raise_warning:
            raise RuntimeWarning(f"File {str(file)} is not a valid JSON file.") from e
        return None
    except AssertionError as e:
        if raise_warning:
            raise RuntimeWarning(f"JSON file {str(file)} does not contain valid metadata.") from e
        return None

    return {k: json_data[k] for k in ["epochs", "resize"]}


print("Starting DFS traversal.")
dfs = []
metakeys = ["resize", "epochs"]
for data_dir in basedir.rglob("benchmark_data"):
    json_files = list(data_dir.rglob("*.json"))
    meta_file = data_dir / "meta.json"
    if meta_file.exists():
        metadata = parse_meta_file(file=meta_file, raise_warning=True)
    else:
        print(f"Found no metadata for data directory {str(data_dir)}.")
        metadata = {"epochs": -1, "resize": -1}

    for data_file in json_files:
        if data_file.name == "meta.json":
            continue
        df = parse_output_file(data_file, raise_warning=False)
        if df is not None:
            df = df.assign(**metadata).set_index(metakeys, append=True)
            dfs.append(df)

big_df = pd.concat(dfs, axis=0)
non_meta_keys = big_df.index.names.difference(metakeys).difference(["idx"])
big_df = big_df.reorder_levels(metakeys + non_meta_keys + ["idx"], axis=0)
big_df.to_pickle(basedir / "data.pkl.gz")
print("Fin.")
