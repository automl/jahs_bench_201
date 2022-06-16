# JAHS-Bench-201

The first collection of surrogate benchmarks for Joint Architecture and Hyperparameter Search, built to support and
facilitate research on multi-objective, cost-aware and (multi) multi-fidelity optimization algorithms.


![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

Please see our [documentation here](https://automl.github.io/jahs_bench_201/).


## Installation

Using pip

```bash
pip install jahs_bench_201
```

### Verify Installation

This is a minimum working example to test if the installation was successful:

```bash
python -m jahs_bench_201_examples.mwe
```

This should randomly sample a configuration and display both the sampled configuration and the result of querying the
surrogate for that configuration.

## Using the benchmark

The API of our benchmark enables users to either query a surrogate model or the tables of performance data, or train a
configuration from our search space from scratch using the same pipeline as was used by our benchmark.
However, users should note that the latter functionality requires the installation of `jahs_bench_201` with the
optional `data_creation` component and its relevant dependencies. The relevant data can be automatically downloaded by
our API.

### Querying the surrogate

```python
# Download the trained surrogate model
from jahs_bench_201.api import Benchmark
b = Benchmark(task="cifar10", kind="surrogate", download=True)

# Query a random configuration
config, results = b.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict

```

### Querying the performance tables

```python
# Download the performance dataset
from jahs_bench_201.api import Benchmark
b = Benchmark(task="cifar10", kind="table", download=True)

# Query a random configuration
config, results = b.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict

```

### Live training a random configuration from scratch

```python
# Initialize the pipeline
from jahs_bench_201.api import Benchmark
b = Benchmark(task="cifar10", kind="live")

# Query a random configuration
config, results = b.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # Only the final epochs' results

```

### Querying the full trajectories

Optionally, the full trajectory of query can be queried by flipping a single flag

```python
config, trajectory = b.random_sample(full_trajectory=True)

print(trajectory)  # A list of dicts
```

## Benchmark data

Information and instructions for downloading the performance dataset used to train our surrogate models can be found [in our documentation](https://automl.github.io/jahs_bench_201/download_dataset).

Information and instructions for downloading our trained surrogate models can also be found [in our documentation](https://automl.github.io/jahs_bench_201/download_surrogate).

## Experiments and Evaluation Protocol

See [our experiments repository](https://github.com/automl/jahs_bench_201_experiments) and our [documentation](https://automl.github.io/jahs_bench_201/evaluation_protocol).

## Leaderboards

We maintain [leaderboards](https://automl.github.io/jahs_bench_201/leaderboards) for several optimization tasks and algorithmic frameworks.
