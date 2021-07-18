import logging
import codecs
from hpbandster.core.master import Master
from hpbandster.core.base_config_generator import base_config_generator as BaseConfigGenerator
from hpbandster.core.base_iteration import BaseIteration
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import utils
from pathlib import Path

class RandomConfigGenerator(BaseConfigGenerator):
    """
    Randomly samples NasBench201 configurations as a list of op-indices.
    """

    def __init__(self, search_space: NasBench201SearchSpace, job_config, config_dir: Path = None, **kwargs):
        super(RandomConfigGenerator, self).__init__(**kwargs)
        if config_dir is not None:
            raise NotImplementedError("Future feature to load configurations from a fixed source of (randomly sampled) "
                                      "configurations.")

        self.search_space = search_space.clone()
        # TODO: Consider adding all fidelity info here
        self.fidelity_info = {}
        # self.fidelity_info = {
        #     "cell channels": self.search_space.CHANNELS,
        #     "cell repeats": self.search_space.CELL_REPEAT,
        #     "image resolution":
        # }
        utils.set_seed(job_config.seed)
        self.job_config = job_config

    def get_config(self, budget):
        # TODO: the parameter budget could be used to dynamically change fidelity of the search space
        self.search_space.sample_random_architecture()
        config = self.search_space.get_op_indices()
        return config, self.fidelity_info


class TabularIteration(BaseIteration):
    """
    A single iteration of a tabular sampling job.
    """

    def _advance_to_next_stage(self, config_ids, losses):
        """
            Allows all sampled configurations to be marked as valid.
        """

        return [True] * len(config_ids)


class TabularSampling(Master):
    def __init__(self, job_config, **kwargs):
        """
        Perform sampling of a given search space in order to build a tabular benchmark. The unique parameters of the
        initializer are wrapped up in the job_config:

        Parameters
        ----------
        n_iters: int
            The number of iterations the sampler should run for. A single iteration of the sampler will sample
            'nsamples' configurations.
        nsamples: int
            The number of configurations to be sampled per iteration of the sampler.
        """

        # TODO: Possibly add functionality to sample different configurations in each iteration or change certain
        #  parameters while sampling the same configurations in each iteration.

        super(TabularSampling, self).__init__(**kwargs)
        self.job_config = job_config
        self.n_iters = job_config.sampler.n_iters
        self.nsamples = job_config.sampler.nsamples

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """
        Returns a TabularIteration.

        Parameters
        ----------
            iteration: int
                the index of the iteration to be instantiated

        Returns
        -------
            TabularIteration: the TabularIteration iteration with the corresponding number of configurations
        """

        return (TabularIteration(HPB_iter=iteration, num_configs=self.nsamples, budgets=[0],
                                  config_sampler=self.config_generator.get_config, **iteration_kwargs))
