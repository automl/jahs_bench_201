import jahs_bench


def run():
    benchmark = jahs_bench.Benchmark(task="cifar10", kind="surrogate", download=True)
    config = benchmark.sample_config()
    results = benchmark(config, nepochs=200)
    print(f"Sampled random configuration: {config}\nResults of query on the surrogate "
          f"model: {results}")


if __name__ == "__main__":
    run()
