# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ARE Environment for OpenEnv.

This module provides OpenEnv integration for the Agents Research Environment (ARE),
which is an event-driven simulation framework for scenarios involving apps and tool calling.

Example:
    >>> from envs.are_env import AREEnv, InitializeAction, TickAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = AREEnv.from_docker_image("are-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> result = env.step(InitializeAction(scenario_path="/path/to/scenario.json"))
    >>> result = env.step(TickAction(num_ticks=5))
    >>> print(result.observation.current_time)
    >>>
    >>> # Cleanup
    >>> env.close()
"""

from .client import AREEnv
from .models import (
    AREAction,
    AREObservation,
    AREState,
    CallToolAction,
    GetStateAction,
    InitializeAction,
    ListAppsAction,
    TickAction,
)

__all__ = [
    "AREEnv",
    "AREAction",
    "AREObservation",
    "AREState",
    "InitializeAction",
    "TickAction",
    "ListAppsAction",
    "CallToolAction",
    "GetStateAction",
]
