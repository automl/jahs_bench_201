import logging
from functools import partial
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
from jahs_bench.core.configspace import joint_config_space
from jahs_bench.surrogate.model import XGBSurrogate

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)


class Benchmark:
    def __init__(self, model_path: Optional[Union[str, Path]],
                 outputs: Optional[Sequence[str]] = None):
        """
        Load up the benchmark for querying. Currently, only the surrogate benchmark is
        supported.
        :param model_path: str or Path-like
            The path to a directory where a pre-trained surrogate model or set of models
            is/are stored.
        :param outputs: List of str or None
            A separate model is used to predict each performance metric. Thus, when
            multiple metrics need to be queried, all corresponding models should be
            loaded up. When this is None, all available models are loaded (default).
            Otherwise, a list of strings can be provided to load the models for a subset
            of all metrics.
        """
        if isinstance(model_path, str):
            model_path = Path(model_path)

        assert model_path is not None, \
            "A path to a directory where a surrogate model was saved must be given."
        assert model_path.exists() and model_path.is_dir()

        if outputs is None:
            outputs = [p.name for p in model_path.iterdir() if p.is_dir()]

        self.surrogates = {}
        for o in outputs:
            self.surrogates[o] = XGBSurrogate.load(model_path / str(o))
        self._call_fn = self._benchmark_surrogate
        # self.surrogate = XGBSurrogate.load(model_path)
        # self._call_fn = partial(self._benchmark_surrogate, surrogate=self.surrogate)

    def __call__(self, config: dict, nepochs: Optional[int] = 200):
        return self._call_fn(config=config, nepochs=nepochs)

    def _benchmark_surrogate(self, config: dict, nepochs: Optional[int] = 200) -> dict:
        features = pd.Series(config).to_frame().transpose()
        features.loc[:, "epoch"] = nepochs

        outputs = {}
        for o, model in self.surrogates.items():
            outputs[o] = model.predict(features)

        outputs: pd.DataFrame = pd.concat(outputs, axis=1)
        return outputs.to_dict()
        # outputs: np.ndarray = surrogate.predict(features).values
        # outputs = outputs.reshape(-1, surrogate.label_headers.size)

        # return {k: outputs[0][i] for i, k in enumerate(surrogate.label_headers.values)}


if __name__ == "__main__":
    conf = joint_config_space.sample_configuration().get_dictionary()
    model_path = Path(__file__).parent.parent / "trained_surrogate_models" / \
                 "thesis_cifar10"
    #     model_path = model_path / Path("../surrogates/full_data").resolve()
    print(f"Attempting to read surrogate model from: {model_path}")
    b = Benchmark(model_path=model_path)
    res = b(config=conf, nepochs=200)
    print(res)
