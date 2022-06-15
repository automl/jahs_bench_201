from pathlib import Path
from typing import Union
from copy import deepcopy

from hpbandster.core.worker import Worker

from jahs_bench.public_api import Benchmark
from jahs_bench_201.lib.core.constants import datasets

from .utils import create_config_space


class JAHS_Bench_wrapper(Worker):
    def __init__(
        self,
        dataset: str,
        model_path: Union[str, Path, list],
        use_default_hps: bool = False,
        use_default_arch: bool = False,
        fidelity: str = "Epochs",
        use_surrogate: bool = True,
        seed: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert dataset in datasets, f"Other benchmarks than {datasets} not supported"

        self.dataset = dataset
        self.use_surrogate = use_surrogate  # TODO: Allow for live_training as well

        self.benchmark_fn = Benchmark(
            model_path=model_path,
            outputs=["valid-acc", "latency", "runtime"]
        )

        self.fidelity = fidelity
        self.default_config, self._joint_config_space = create_config_space(
            use_default_arch=use_default_arch,
            use_default_hps=use_default_hps,
            fidelity=fidelity,
            seed=seed
        )

    def compute(
        self,
        config,
        budget,
        working_directory,
        *args,
        **kwargs
    ):
        query_config = deepcopy(self.default_config)
        query_config.update(config)
        if self.fidelity == "Epochs":
            nepochs = int(budget)
        else:
            query_config[self.fidelity] = budget
            nepochs = 200
        if "Epochs" in query_config:
            query_config.pop("Epochs")

        results = self.benchmark_fn(
            config=query_config,
            nepochs=nepochs
        )

        return({
                    'loss': float(100 - float(list(results[("valid-acc", "valid-acc")].values())[-1])),
                    'info': dict(
                        cost=float(list(results[("runtime", "runtime")].values())[-1]),
                        latency=float(list(results[("latency", "latency")].values())[-1])
                    )
                })

    @property
    def joint_config_space(self):
        return self._joint_config_space


if __name__ == "__main__":

    model_path = Path(__file__).parent.parent.parent.parent / "JAHS-Bench-MF" / "trained_surrogate_models" / "thesis_cifar10"
    b = JAHS_Bench_wrapper(
        dataset="cifar10",
        model_path=model_path,
        use_default_hps=False,
        use_default_arch=True,
        fidelity="N",
        seed=15,
    )
    print(b.joint_config_space.get_hyperparameters_dict())
    print(b.default_config)
