# Querying JAHS-Bench-201

## Evaluating Configurations with The Surrogate

```python
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", download=True)

# Query a random configuration
config = benchmark.sample_config()
results = benchmark(config)

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict
```

## Querying the Full Trajectories

Optionally, the full trajectory of query can be queried by flipping a single flag

```python
config, trajectory = benchmark.random_sample(full_trajectory=True)

print(trajectory)  # A list of dicts
```




## More Evaluation Options

The API of our benchmark enables users to either query a surrogate model (the default) or the tables of performance data, or train a
configuration from our search space from scratch using the same pipeline as was used by our benchmark.
However, users should note that the latter functionality requires the installation of `jahs_bench_201` with the
optional `data_creation` component and its relevant dependencies. The relevant data can be automatically downloaded by
our API.


### Querying the Surrogate

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

### Querying the Performance Tables

```python
# Download the performance dataset
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="table", download=True)

# Query a random configuration
config, results = benchmark.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # A dict

```

### Live Training a Random Configuration from Scratch

```python
# Initialize the pipeline
import jahs_bench

benchmark = jahs_bench.Benchmark(task="cifar10", kind="live")

# Query a random configuration
config, results = benchmark.random_sample()

# Display the outputs
print(f"Config: {config}")  # A dict
print(f"Result: {results}")  # Only the final epochs' results

```
