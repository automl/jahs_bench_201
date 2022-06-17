# Querying JAHS-Bench-201

## Evaluating Configurations with the Surrogate

```python
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", download=True)

# Query a random configuration
config = benchmark.sample_config()
results = benchmark(config, nepochs=200)

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict of dicts, indexed first by epoch and then by metric name
```

## Querying the Full Trajectories

Optionally, the full trajectory of a query can be queried by flipping a single flag

```python
config = benchmark.sample_config()
trajectory = benchmark(config, nepochs=200, full_trajectory=True)

print(trajectory)  # A dict of dicts
```




## More Evaluation Options

The API of our benchmark enables users to either query a surrogate model (the default) or the tables of performance data, or train a
configuration from our search space from scratch using the same pipeline as was used by our benchmark.
However, users should note that the latter functionality requires the installation of `jahs_bench_201` with the
optional `data_creation` component and its relevant dependencies. The relevant data can be automatically downloaded by
our API.

Users may switch between querying the surrogate model, the performance dataset, or a live training of a configuration
by passing one of the strings `surrogate` (default), `table` or `live` to the parameter `kind` when initializing the
`Benchmark` object, as:

```python
benchmark_surrogate = jahs_bench.Benchmark(task="cifar10", kind="surrogate", download=True)
benchmark_tabular = jahs_bench.Benchmark(task="cifar10", kind="table", download=True)
benchmark_live = jahs_bench.Benchmark(task="cifar10", kind="live", download=True)
```

Setting the flag `download` to True allows the API to automatically fetch all the relevant data files over the internet.
This includes the surrogate models, the performance dataset DataFrame objects, and the task datasets and their splits,
depending on whether `kind` was set to `surrogate`, `table` or `live`.

__Note:__ Before using the option `kind="live"`, it is necessary to ensure that JAHS-Bench-201 has been installed with
the extra dependencies required for live training. This can be done using the following command:

```bash
pip install "jahs_bench[data_creation] @ git+https://github.com/automl/jahs_bench_201.git"
```

## More Examples

### Querying the Performance Tables

```python
# Download the performance dataset
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="table", download=True)

# Query a random configuration
config = benchmark.sample_config()
results = benchmark(config, nepochs=200)

# Display the outputs
print(f"Config: {config}")
print(f"Result: {results}")

```

### Querying the Performance Tables with Full Trajectory

```python
# Download the performance dataset
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="table", download=True)

# Query a random configuration
config = benchmark.sample_config()
trajectory = benchmark(config, nepochs=200, full_trajectory=True)

# Display the outputs
print(f"Config: {config}")
print(f"Result: {trajectory}")
```

### Live Training a Random Configuration from Scratch with Full Trajectory

```python
# Initialize the pipeline
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="live", download=True)

# Query a random configuration
config = benchmark.sample_config()
trajectory = benchmark(config, nepochs=200, full_trajectory=True)

# Display the outputs
print(f"Config: {config}")
print(f"Result: {trajectory}")
```
