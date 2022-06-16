from jahs_bench import api


def run():
    b = api.Benchmark(task="cifar10", kind="surrogate", download=True)
    config, results = b.random_sample()
    print(f"Sampled random configuration: {config}\nResults of query on the surrogate "
          f"model: {results}")

if __name__ == "__main__":
    run()
