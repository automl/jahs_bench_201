import jahs_bench


def run():
    benchmark = jahs_bench.Benchmark(task="cifar10", kind="surrogate", download=True)
    config, results = benchmark.random_sample()
    print(f"Sampled random configuration: {config}\nResults of query on the surrogate "
          f"model: {results}")


if __name__ == "__main__":
    run()
