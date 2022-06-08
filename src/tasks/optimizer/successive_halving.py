"""
Official implementation of RandomSearch from the HpBandster Repo extended to SH:
https://github.com/automl/HpBandSter/blob/master/hpbandster/optimizers/randomsearch.py
"""

import os
import time
import math
import copy
import logging

import ConfigSpace
import numpy as np

import ConfigSpace as CS

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving as SH
from hpbandster.optimizers.config_generators.random_sampling import RandomSampling as RS

from wrapper.utils import fidelities, get_diagonal


class SuccessiveHalving(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=1, max_budget=1, fidelity="Epochs",
                 **kwargs
                 ):
        """
                Implements a random search across the search space for comparison.
        Parameters
        ----------
        configspace: ConfigSpace object
            valid representation of the search space
        eta : float
            In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        budget : float
            budget for the evaluation
        """

        if configspace is None:
            raise ValueError("You have to provide a valid ConfigSpace object")

        cg = RS(configspace=configspace)

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        self.fidelity = fidelity
        if isinstance(fidelity, list):
            self.budgets = get_diagonal(fidelity, eta=eta)
            self.max_SH_iter = len(self.budgets)
            self.eta = 2
        else:
            _fidelity = fidelities[self.fidelity]
            if isinstance(_fidelity, ConfigSpace.OrdinalHyperparameter):
                self.budgets = _fidelity.sequence
                self.max_SH_iter = len(self.budgets)
                self.eta = 2
            else:
                self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
                self.budgets = max_budget * np.power(eta,
                                                     -np.linspace(self.max_SH_iter - 1, 0,
                                                                  self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """
        Returns a SH iteration with only evaluations on the biggest budget

        Parameters
        ----------
            iteration: int
                the index of the iteration to be instantiated

        Returns
        -------
            SuccessiveHalving: the SuccessiveHalving iteration with the
                corresponding number of configurations
        """

        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket

        n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        if isinstance(self.fidelity, list):
            return (
                SH(HPB_iter=iteration, num_configs=ns,
                   budgets=self.budgets[(-s - 1):],
                   config_sampler=self.config_generator.get_config,
                   **iteration_kwargs)
            )
        else:
            return (
                SH(HPB_iter=iteration, num_configs=ns,
                      budgets=self.budgets[(-s-1):],
                      config_sampler=self.config_generator.get_config,
                      **iteration_kwargs)
                    )
