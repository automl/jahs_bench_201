# Introduction and Installation

JAHS-Bench-201 is the first collection of surrogate benchmarks for Joint Architecture and Hyperparameter Search (JAHS), built to also support and
facilitate research on multi-objective, cost-aware and (multi) multi-fidelity optimization algorithms.


To install using pip run

```bash
pip install git+https://github.com/automl/jahs_bench_201.git
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
