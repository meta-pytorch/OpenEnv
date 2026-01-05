# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SUMO-RL Environment for OpenEnv.

This module provides OpenEnv integration for traffic signal control using
SUMO (Simulation of Urban MObility) via the SUMO-RL library.

Example:
    >>> from envs.sumo_rl_env import SumoRLEnv, SumoAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = SumoRLEnv.from_docker_image("sumo-rl-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> result = env.step(SumoAction(phase_id=1))
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import SumoRLEnv
from .models import SumoAction, SumoObservation, SumoState

__all__ = ["SumoRLEnv", "SumoAction", "SumoObservation", "SumoState"]
