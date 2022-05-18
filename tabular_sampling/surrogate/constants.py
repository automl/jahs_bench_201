from enum import Enum
import logging
import yacs.config as config
_log = logging.getLogger(__name__)

class RegressionLossFuncTypes(Enum):
    squared_error = "reg:squarederror"
    custom = "reg:custom"

#  The default configuration for a squared error loss function.
default_se_loss_config = config.CfgNode()
default_se_loss_config.multioutput = "raw_values"

#  The default configuration for a custom exponential boundary loss function used as a
#  lower bound.
root = config.CfgNode()
root.y_lim = 0.
root.argmin = 0.5
root.c = None
root.min_val = 1e-6
root.multioutput = "raw_values"
default_exp_lower_bound_loss_config = root

#  The default configuration for a custom exponential boundary loss function used as an
#  upper bound.
root = config.CfgNode()
root.y_lim = 1.
root.argmin = 0.5
root.c = None
root.min_val = 1e-6
root.multioutput = "raw_values"
default_exp_upper_bound_loss_config = root

class default_base_loss_configs(Enum):
    exp_lower_bound = default_exp_lower_bound_loss_config
    exp_upper_bound = default_exp_upper_bound_loss_config
    se = default_se_loss_config

#  The default configuration for a custom loss function that mixes multiple loss
#  functions into a weighted average loss.
root = config.CfgNode()
root.funcs = ["exp_lower_bound", "exp_upper_bound", "se"]
root.params = [default_base_loss_configs[f].value.clone() for f in root.funcs]
root.weights = [1./3.] * 3
default_mixed_loss_config = root

# The default configuration for the

# The default configuration for a surrogate pipeline
default_pipeline_config = config.CfgNode()
default_pipeline_config.loss = RegressionLossFuncTypes.custom.value
default_pipeline_config.loss_params = default_mixed_loss_config.clone()
default_pipeline_config.config_dir = None
del root
