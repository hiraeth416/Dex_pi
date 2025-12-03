import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


config = _config.get_config("pi05")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = libero_policy.make_libero_example()
result = policy.infer(example)
import ipdb; ipdb.set_trace()

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)