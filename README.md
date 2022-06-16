# JAHS-Bench-201

The first collection of surrogate benchmarks for Joint Architecture and Hyperparameter Search, built to support and
facilitate research on multi-objective, cost-aware and (multi) multi-fidelity optimization algorithms.


![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

Please see our [documentation here](https://automl.github.io/jahs_bench_201/).


## Installation

Using pip

```bash
pip install jahs_bench
```

Optionally, you can download the data required to use the surrogate benchmark ahead of time with
```bash
python -m jahs_bench.download --target surrogates
```

To test if the installation was successful, you can, e.g, run a minimal example with
```bash
python -m jahs_bench_examples.minimal
```
This should randomly sample a configuration, and display both the sampled configuration and the result of querying the
surrogate for that configuration.

## Using the Benchmark

```python
# Download the trained surrogate model
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="surrogate", download=True)

# Query a random configuration
config, results = benchmark.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict

```

### More API Options

The API of our benchmark enables users to either query a surrogate model or the tables of performance data, or train a
configuration from our search space from scratch using the same pipeline as was used by our benchmark.
However, users should note that the latter functionality requires the installation of `jahs_bench_201` with the
optional `data_creation` component and its relevant dependencies. The relevant data can be automatically downloaded by
our API. See our [documentation](https://automl.github.io/jahs_bench_201/usage) for details.

## Benchmark Data

We provide [documentation for the performance dataset](https://automl.github.io/jahs_bench_201/performance_dataset) used to train our surrogate models and [further information on our surrogate models](https://automl.github.io/jahs_bench_201/surrogate).


## Experiments and Evaluation Protocol

See [our experiments repository](https://github.com/automl/jahs_bench_201_experiments) and our [documentation](https://automl.github.io/jahs_bench_201/evaluation_protocol).

## Leaderboards

We maintain [leaderboards](https://automl.github.io/jahs_bench_201/leaderboards) for several optimization tasks and algorithmic frameworks.
