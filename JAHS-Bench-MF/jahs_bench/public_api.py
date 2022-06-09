import logging
from functools import partial
from pathlib import Path
from typing import Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd
from jahs_bench.core.configspace import joint_config_space
from jahs_bench.surrogate.model import XGBSurrogate

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


class Benchmark:
    _call_fn = None
    _surrogates = None
    _table = None

    def __init__(self, model_path: Optional[Union[str, Path]] = None,
                 table_path: Optional[Union[str, Path]] = None,
                 outputs: Optional[Sequence[str]] = None):
        """
        Load up the benchmark for querying. Currently, only the surrogate benchmark is
        supported.
        :param model_path: str or Path-like
            The path to a directory where a pre-trained surrogate model or set of models
            is/are stored.
        :param model_path: str or Path-like
            The path to a file where the performance dataset of the benchmark is stored.
        :param outputs: List of str or None
            A separate model is used to predict each performance metric. Thus, when
            multiple metrics need to be queried, all corresponding models should be
            loaded up. When this is None, all available models are loaded (default).
            Otherwise, a list of strings can be provided to load the models for a subset
            of all metrics.
        """

        if isinstance(model_path, str):
            model_path = Path(model_path)

        if isinstance(table_path, str):
            model_path = Path(model_path)

        if model_path is not None:
            self._load_surrogate(model_path, outputs)
        elif table_path is not None:
            self._load_table(table_path, outputs)
        else:
            raise RuntimeError("A path to either a directory where a surrogate model "
                               "or a file containing the performance dataset must be "
                               "given.")

    def _load_surrogate(self, model_path: Optional[Union[str, Path]] = None,
                        outputs: Optional[Sequence[str]] = None):
        assert model_path.exists() and model_path.is_dir()

        if outputs is None:
            outputs = [p.name for p in model_path.iterdir() if p.is_dir()]

        self._surrogates = {}
        for o in outputs:
            self._surrogates[o] = XGBSurrogate.load(model_path / str(o))
        self._call_fn = self._benchmark_surrogate

    def _load_table(self, table_path: Optional[Union[str, Path]] = None,
                    outputs: Optional[Sequence[str]] = None):
        assert table_path.exists() and table_path.is_file()

        table = pd.read_pickle(table_path)
        level_0_cols = ["features", "labels"]
        features: list = joint_config_space.get_hyperparameter_names() + ["epoch"]

        if table["features"].columns.intersection(features).size != len(features):
            raise ValueError(f"The given performance dataset at {table_path} could not "
                             f"be resolved against the known search space consisting of "
                             f"the parameters {features}")
        features = table["features"].columns
        labels: pd.Index = table["labels"].columns

        if outputs is not None:
            # Attempt to convert the sequence of outputs into a list
            outputs = list(outputs) if not isinstance(outputs, list) \
                else [outputs] if isinstance(outputs, str) else outputs

            if labels.intersection(outputs).size != len(outputs):
                raise ValueError(f"The set of outputs specified for the performance "
                                 f"dataset {outputs} must be a subset of all available "
                                 f"outputs: {labels.tolist()}.")

            # Drop all unnecessary outputs
            table.drop([("labels", l) for l in labels.difference(outputs)], axis=1,
                       inplace=True)

        # TODO: Deal with the issue of the index being possibly non-unique, since there
        #  are no guarantees that a configuration wasn't sampled twice.
        # Make the DataFrame indexable by configurations
        # table.set_index(table[["features"]].columns.tolist(), inplace=True)
        # table.index.names = features.tolist()
        # table = table.droplevel(0, axis=1)
        self._table = table
        self._call_fn = self._benchmark_tabular

    def __call__(self, config: dict, nepochs: Optional[int] = 200, **kwargs):
        return self._call_fn(config=config, nepochs=nepochs)

    def _benchmark_surrogate(self, config: dict, nepochs: Optional[int] = 200) -> dict:
        features = pd.Series(config).to_frame().transpose()
        features.loc[:, "epoch"] = nepochs

        outputs = {}
        for o, model in self._surrogates.items():
            outputs[o] = model.predict(features)

        outputs: pd.DataFrame = pd.concat(outputs, axis=1)
        return outputs.to_dict()

    def _benchmark_tabular(self, config: dict, nepochs: Optional[int] = 200,
                           suppress_keyerror: bool = False) -> dict:
        raise NotImplementedError("The functionality for directly querying the tabular "
                                  "performance dataset is still under construction.")
        assert self._table is not None, "No performance dataset has been loaded into " \
                                        "memory - a tabular query cannot be made."
        query = config.copy()
        query["epoch"] = nepochs
        query = tuple((query[k] for k in self._table.index.names))
        try:
            output = self._table.loc[query].to_dict(orient="index")
            output = list(output.values())[0]
        except KeyError as e:
            _log.debug(f"Registered a key-error while querying the performance dataset "
                       f"for the configuration: {config} at {nepochs} epochs. The "
                       f"constructed query was: {query}.")
            if suppress_keyerror:
                output = {}
            else:
                raise KeyError(f"Could not find any entries for the config {config} at "
                               f"{nepochs} epochs.") from e

        return output

    def random_sample(self, random_state: Optional[\
                      Union[int, np.random.RandomState]] = None) -> Tuple[dict, dict]:
        """ Randomly query the benchmark for a configuration. If a tabular benchmark has
        been loaded, a sample from the set of known configurations is queried. Otherwise,
        a random configuration is sampled from the search space and queried on the
        surrogate benchmark. An optional seed for initializing an RNG or a
        pre-initialized RNG may be passed in `random_state` for reproducible queries. """

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        if self._table is not None:
            index = random_state.choice(self._table.index)
            row = self._table.loc[index]
            query = row["features"].to_dict()
            output = row["labels"].to_dict()

            return query, output

            # TODO: Reinstate the functionality to query the table itself for consistency
            #  once the issue of non-unique indices has been fixed
            # query = random_state.choice(self._table.index)
            # query = {self._table.index.names[i]: query[i] for i in range(len(query))}

            # Quite convoluted and redundant, but this helps maintain consistency.
            # nepochs = query.pop("epoch")
            # output = self(config=query, nepochs=nepochs)
            # return {**query, **{"epoch": nepochs}}, output
        else:
            raise NotImplementedError("Random sampling has been implemented only for "
                                      "the performance dataset.")


if __name__ == "__main__":
    conf = joint_config_space.sample_configuration().get_dictionary()
    model_path = Path(__file__).parent.parent / "trained_surrogate_models" / \
                 "thesis_cifar10"
    #     model_path = model_path / Path("../surrogates/full_data").resolve()
    print(f"Attempting to read surrogate model from: {model_path}")
    b = Benchmark(model_path=model_path)
    res = b(config=conf, nepochs=200)
    print(res)
