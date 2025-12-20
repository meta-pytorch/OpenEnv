# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application that exposes a generic Gymnasium environment."""

import os

from core.env_server import create_app

from ..models import GymAction, GymObservation
from .gymnasium_environment import GymnasiumEnvironment
import yaml

# Environment configuration via environment variables
env_id = os.getenv("GYM_ENVIRONMENT_ID", "MountainCarContinuous-v0")
render_mode = os.getenv("GYM_RENDER_MODE", "rgb_array")

max_steps_str = os.getenv("GYM_MAX_STEPS")
max_steps = int(max_steps_str) if max_steps_str else 1000

seed_str = os.getenv("GYM_SEED")
seed = int(seed_str) if seed_str else None
yaml_param_file_path = os.getenv("ADDITIONAL_PARAMETERS_YAML_FILE")
additional_params = {}

# Load additional parameters from YAML if file path is provided
if yaml_param_file_path and os.path.exists(yaml_param_file_path):
    with open(yaml_param_file_path, "r") as f:
        additional_params = yaml.safe_load(f)

# Create the environment instance
env = GymnasiumEnvironment(
    env_id=env_id,
    render_mode=render_mode,
    max_steps=max_steps,
    seed=seed,
    **additional_params,
)

# Create the FastAPI app with web interface and README integration
app = create_app(
    env,
    GymAction,
    GymObservation,
    env_name=env_id.lower().replace("-", "_"),
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
