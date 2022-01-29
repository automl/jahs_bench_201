""" Very simple script for generating a reproducible set of hyper-parameters for the surrogate. """

from pathlib import Path
import json
import pandas as pd
import sys
from tabular_sampling.surrogate import xgb
from sklearn.model_selection import ParameterSampler
import pickle
import numpy as np

# Randomly generated seed
rng = np.random.RandomState(312828812)
nsamples = 10
sampler = ParameterSampler(xgb.XGBSurrogate._hpo_search_space, n_iter=nsamples, random_state=rng)
params = list(sampler)
with open("/home/archit/thesis/nashpo_benchmarks/tabular_sampling/surrogate/candidate_hps.json", "w") as fp:
    pickle.dump(params, fp)