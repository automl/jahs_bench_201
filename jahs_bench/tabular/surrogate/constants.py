import logging
from enum import Enum

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

# The default configuration for a custom exponential boundary loss function used as an
# upper bound.
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


# The default configuration for a custom loss function that mixes multiple loss functions
# into a weighted average loss.
# 'funcs' and 'params' here correspond to lists of equal length containing the names of
# the loss functions and their corresponding keyword-arguments, respectively. The
# 'weights' are the mixture weights for the corresponding functions. The number of
# functions used here is flexible.
root = config.CfgNode()
root.funcs = ["exp_lower_bound", "exp_upper_bound", "se"]
root.params = [default_base_loss_configs[f].value.clone() for f in root.funcs]
root.weights = [1. / 3.] * 3
default_mixed_loss_config = root

# The default configuration for the transformations to be applied to the target variable
# This may be either a single transformation or a chain of transformations.
# In the case of a single transformation, set the key 'transform' to the name of the
# desired transformation and 'params' to the relevant keyword-arguments.
# In order to define a chain of transformations, set 'transform' to "chain" and pass
# a dictionary to 'params' containing two keys: "funcs" and "params", corresponding to
# lists of equal length containing the transformation names and keyword-arguments,
# respectively.
root = config.CfgNode()
root.transform = "chain"
root.params = config.CfgNode({
    "funcs": ["MinMax", "custom:inverse_sigmoid"],
    "params": [
        config.CfgNode({"feature_range": (1e-6, 1. - 1e-6)}),
        config.CfgNode({"k": 1.})
    ]
})
default_target_config = root

# The default configuration for a surrogate pipeline
# Regarding the loss function, instead of using a mixture of losses as here, a single
# loss function name can be passed to 'loss' and the corresponding keyword-arguments to
# 'loss_params'. 'loss_params' (or 'params' in the case of a mixture of losses) can also
# be the name of a .yaml file that resides in 'config_dir' in order to load the
# corresponding values from there.
default_pipeline_config = config.CfgNode()
default_pipeline_config.loss = RegressionLossFuncTypes.custom.value
default_pipeline_config.loss_params = default_mixed_loss_config.clone()
default_pipeline_config.config_dir = None
default_pipeline_config.target_config = default_target_config
default_pipeline_config.set_new_allowed(False)
default_pipeline_config.freeze()
del root
