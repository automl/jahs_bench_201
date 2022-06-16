# Using JAHS-Bench-201


## Querying the surrogate

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

## Querying the performance tables

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

## Live training a random configuration from scratch

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

## Querying the full trajectories

Optionally, the full trajectory of query can be queried by flipping a single flag

```python
config, trajectory = benchmark.random_sample(full_trajectory=True)

print(trajectory)  # A list of dicts
```
