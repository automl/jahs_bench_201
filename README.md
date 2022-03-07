# JAHS-Bench-MF

TODO: Sentence describing JAHS-Bench-MF. (+ Reference documentation page once it exists)

Features:

- TODO

Soon-to-come Features:

- TODO

![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-informational)
[![License](TODO)](LICENSE)

## Installation

Using pip

```bash
git clone https://github.com/automl/jahs_bench_mf.git
cd jahs_bench_mf
pip install .
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
model_path = "jahs_bench_mf/JAHS-Bench-MF/surrogates/full_data"
b = Benchmark(model_path=model_path)

# Generate a random configuration
from jahs_bench.lib.configspace import joint_config_space
conf = joint_config_space.sample_configuration().get_dictionary()

# Query the configuration
res = b(config=conf, nepochs=200)
print(res)

```

## Contributing

Please see our guidelines and guides for contributors at [CONTRIBUTING.md](CONTRIBUTING.md).
