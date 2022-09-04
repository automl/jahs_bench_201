from jahs_bench.api import Benchmark
from jahs_bench.lib.core.configspace import joint_config_space


if __name__ == "__main__":

    DATASET = "cifar10"
    MODEL_PATH = "."
    NEPOCHS = 200
    N_ITERATIONS = 100

    benchmark = Benchmark(
            task=DATASET,
            save_dir=MODEL_PATH,
            kind="surrogate",
            download=True
        )

    # Random Search
    configs = []
    results = []
    for it in range(N_ITERATIONS + 1):
        # Use benchmark ConfigSpace object to sample a random configuration.
        config = joint_config_space.sample_configuration().get_dictionary()
        # Alternatively, define configuration as a dictionary.
        # config = {
        #     'Optimizer': 'SGD',
        #     'LearningRate': 0.1,
        #     'WeightDecay': 5e-05,
        #     'Activation': 'Mish',
        #     'TrivialAugment': False,
        #     'Op1': 4,
        #     'Op2': 1,
        #     'Op3': 2,
        #     'Op4': 0,
        #     'Op5': 2,
        #     'Op6': 1,
        #     'N': 5,
        #     'W': 16,
        #     'Resolution': 1.0,
        # }
        result = benchmark(config, nepochs=NEPOCHS)

        configs.append(config)
        results.append(100 - float(result[NEPOCHS]["valid-acc"]))

    incumbent_idx = min(range(len(results)), key=results.__getitem__)
    incumbent = configs[incumbent_idx]
    incumbent_value = results[incumbent_idx]
    print(f"Incumbent: {incumbent} \n Incumbent Value: {incumbent_value}")
