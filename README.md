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

### Verify Installation

This is a minimum working example to test if the installation was successful:

```
cd jahs_bench_mf
python JAHS-Bench-MF/jahs_bench/public_api.py
```

This should randomly sample a configuration and display the result of querying the surrogate for that configuration.

## Data

The trained surrogate models can be downloaded either automatically using our API or from ...

The performance dataset and the train/valid/test splits used to train our surrogate models can be downloaded either
through our API or from ...

Currently, we share all our data in the form of Pandas DataFrames which are very efficient for handling large tables of
data, stored as compressed pickle files using pickle protocol 4. The current hosting solution is a transitory one as we
work towards setting up a more robust solution using [Figshare+](https://figshare.com/), which provides perpetual data
storage guarantees, a DOI and a web API for querying the dataset as well as the metadata. Additionally, we are aware of
the inherent isssues with sharing pickle files and therefore are investigating the most appropriate data format.
Current candidates include CSV, HDF5 and Feather.

## Usage

The API of our benchmark enables users to either query a surrogate model or the tables of performance data, or train a
configuration from our search space from scratch using the same pipeline as was used by our benchmark.
However, users should note that the latter functionality requires the installation of `jahs_bench_201` with the
optional `data_creation` component and its relevant dependencies.

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

## Further documentation

More detailed documentation can be found **TODO**
## TODO

Archit:
* `pip install neural-pipeline-search` for HPO of surrogate (Archit) -- include in documentation
* Reference section of Readme/Documentation explaining how to install optional components in the section "Usage"


Danny
* `python -m jahs_bench_201.download_surrogates` and `download=True` as default (danny)
* write leaderboards text
* Placeholders Documentation
* https://github.com/automl/jahs_bench_201_experiments mention
* Go over README
* Experiments repo fill

Maciej
* We provide code to use JAHS-Bench-201 and follow our evaluation protocols (latter Maciej)
* leaderboard entry (Maciej)

Open
* Documentation at https://automl.github.io/jahs_bench_201/
* Put on pypi
* Fix: The 'automl/jahs_bench_201' repository doesn't contain the 'TODO' path in 'main'.


From appendix copied:


    * Dataset Documentation \todo{Mostly requirements from NeurIPS}
        * URL to the dataset and its metadata (must be structured; the guidelines mention using a web standard like schema.org or DCAT for this)
        * Instructions on accessing the dataset
        * Datasheets - for us, this would be broad properties such as the format, disk space requirements, table dimensions, other metadata. \todo{Is there a standard framework that we can use?}
        * License and author statement, ethical/responsible use guidelines. Author statement that they bear all responsibility in case of violation of rights, etc., and confirmation of the data license.
        * Recommended from the guidelines: accountability framework
        * Hosting, licensing and maintenance plan. Clarify long-term preservation.
        * Highly recommended: a persistent dereferenceable identifier, e.g. DOI or prefix from identifiers.org. (We already are on GitHub)
        * Compute usage
        * Ethics statement
    * API/Git Repo [Referenced by section 1]
        * Detailed instructions for using the dataset.
        * Minimum Working Example(s)
        * Reproducibility documentation - instructions, data, code.
