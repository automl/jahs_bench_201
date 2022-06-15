# JAHS-Bench-201

The first collection of surrogate benchmarks for Joint Architecture and Hyperparameter Search, built to support and facilitate research on multi-objective, cost-aware and (multi) multi-fidelity optimization algorithms.

Features:
- TODO

![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](TODO)](LICENSE)

## Installation

Using pip

```bash
pip install jahs_bench_201
```


## Usage

### Verify Installation

Here is a minimum working example to test whether or not the installation succeeded:

```
cd jahs_bench_mf
python JAHS-Bench-MF/jahs_bench/public_api.py
```

This should randomly sample a configuration and display the result of querying for that configuration on the surrogate.

### Querying the surrogate

```python
# Load the trained surrogate model
from jahs_bench.public_api import Benchmark

model_path = "jahs_bench_mf/JAHS-Bench-MF/surrogates/thesis_cifar10"
b = Benchmark(model_path=model_path)

# Generate a random configuration
from jahs_bench_201.lib.core.configspace import joint_config_space

conf = joint_config_space.sample_configuration().get_dictionary()

# Query the configuration
res = b(config=conf, nepochs=200)
print(res)

```

**Caution:** The repeatability of the surrogate model's predictions is still under investigation and carries no
guarantees.

## Contributing

Please see our guidelines and guides for contributors at [CONTRIBUTING.md](CONTRIBUTING.md).


## TODO

* share data with others (Archit) and say we upload to figshare lifetime support later, plan/left-todo, refer to figshare for the future)
* reorder code (Archit)
* We provide code to use JAHS-Bench-201 and follow our evaluation protocols (latter Maciej)
* `python -m jahs_bench_201.download_surrogates` and `download=True` as default
* leaderboards (Maciej)
* Docstrings
* Documentation at https://automl.github.io/jahs_bench_201/
* Placeholders Documentation
* `pip install neural-pipeline-search` for HPO of surrogate (Archit)
* Add dataset licenses to NOTICE (Archit)
* https://github.com/automl/jahs_bench_201_experiments mention
