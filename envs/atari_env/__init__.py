# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari Environment for OpenEnv.

This module provides OpenEnv integration for Atari 2600 games via the
Arcade Learning Environment (ALE).

Example:
    >>> from envs.atari_env import AtariEnv, AtariAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = AtariEnv.from_docker_image("atari-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> result = env.step(AtariAction(action_id=2))  # UP
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import AtariEnv
from .models import AtariAction, AtariObservation, AtariState

__all__ = ["AtariEnv", "AtariAction", "AtariObservation", "AtariState"]
