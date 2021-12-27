import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

_log = logging.getLogger(__name__)


# def generate_cv_splits(dataset_size: int, num_splits: int) -> List[Tuple[List[int], List[int]]]:
#     """ Generates cross-validation splits for any dataset of a given size. The number of splits 'num_splits' to be
#     generated must be greater than or equal to 2. This returns a list of size 'num_splits', such that each element in
#     the list is itself a tuple of two lists. Each such tuple corresponds to one split of the cross-validation and the
#     lists correspond to the data indices to be used for training and validation respectively in that split. """
#
#     assert num_splits >= 2, f"The number of splits for generating cross-validation index splits must be greater than " \
#                             f"or equal to 2, was given {num_splits}."
#
#     indices = np.arange(dataset_size)
#     split_size = dataset_size // num_splits
#     splits = np.array_split(indices, split_size)
#     cv_splits = [(np.concatenate(splits[:i] + splits[i+1:]), splits[i]) for i in range(num_splits)]
#
#     return cv_splits

