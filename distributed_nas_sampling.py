import codecs
import json
import logging
import os
import sys
import time
from pathlib import Path

from naslib.search_spaces import (
    NasBench201SearchSpace,
)

from naslib.utils.logging import log_every_n_seconds, log_first_n
from naslib.utils import utils, setup_logger, logging as naslib_logging

import torch
import numpy as np
from hpbandster.core import nameserver as hpns
from hpbandster.core.nameserver import nic_name_to_host
from hpbandster.core.result import json_result_logger
from naslib.tabular_sampling import ModelTrainer, RandomConfigGenerator, TabularSampling

# Read args and config, setup logger
start_time = time.time()
config = utils.get_config_from_args()

# TODO: Introduce two loggers. One for putting together streams from multiple workers and one for writing data.
logger = setup_logger(config.save + "/log.log")
#logger.setLevel(logging.INFO)   # default DEBUG is very verbose

utils.log_args(config)
# Extra params for this script should be specified as optional parameters towards the end under the heading "sampler",
# e.g. "sampler.process_offset 0"

# process_offset 0 is the master, process_offset > 0 are workers, process_offset = -1 implies a single master which
# also runs a worker
jobid = config.sampler.jobid
proc_offset = config.sampler.process_offset

# Set up search space
network_depth = config.sampler.network_depth
network_width = config.sampler.network_width

network_depth_map = {1: 2, 2: 3, 3: 5}
network_width_map = {1: (8, 8, 16), 2: (8, 16, 32), 3: (16, 32, 64)}

NasBench201SearchSpace.CELL_REPEAT = network_depth_map[network_depth]
NasBench201SearchSpace.CHANNELS = network_width_map[network_width]
search_space = NasBench201SearchSpace()

working_dir = Path(config.save) / jobid
working_dir.mkdir(parents=True, exist_ok=True)
host = nic_name_to_host('lo')

if proc_offset > 0:
    # Initialize worker process
    w = ModelTrainer(run_id=jobid, search_space=search_space, seed=config.seed, job_config=config,
                     host=host)
    w.load_nameserver_credentials(working_directory=working_dir)
    w.run(background=False)
    exit(0)

# Initialize Master process
result_logger = json_result_logger(directory=working_dir, overwrite=False)

NS = hpns.NameServer(run_id=jobid, host=host, port=0, working_directory=working_dir)
ns_host, ns_port = NS.start()

config_generator = RandomConfigGenerator(search_space=search_space, job_config=config)
sampler = TabularSampling(job_config=config, run_id=jobid, config_generator=config_generator, nameserver=ns_host,
                          nameserver_port=ns_port, working_directory=working_dir, logger=logger,
                          result_logger=result_logger)

_ = sampler.run()

# logger.info("Start training")
#
#
# if not os.path.exists(config.save):
#     os.makedirs(config.save)
#
# with codecs.open(os.path.join(config.save, 'errors.json'), 'w', encoding='utf-8') as file:
#     json.dump(errors_dict, file, separators=(',', ':'), indent=4)
#
# logger.info("Exiting.")
