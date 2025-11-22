# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv Environments
====================

This package contains all environment implementations for OpenEnv.

Each environment provides:
- An environment client class (e.g., CodingEnv, AtariEnv)
- Action and Observation data classes
- Server implementations for the HTTP API

Auto Classes
------------
The AutoEnv and AutoAction classes provide a HuggingFace-style API for
automatically selecting the correct environment and action types based on
environment names.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # Automatically detect and create environment from name
    >>> client = AutoEnv.from_name("coding-env")
    >>>
    >>> # Get the corresponding Action class
    >>> CodeAction = AutoAction.from_name("coding-env")
    >>>
    >>> # Use them together
    >>> result = client.reset()
    >>> action = CodeAction(code="print('Hello, AutoEnv!')")
    >>> step_result = client.step(action)
    >>> client.close()

Direct Imports
--------------
You can also import specific environment classes directly:

    >>> from envs.coding_env import CodingEnv, CodeAction
    >>> from envs.echo_env import EchoEnv, EchoAction
    >>> from envs.git_env import GitEnv, GitAction
    >>> # ... etc

List Available Environments
---------------------------
To see all available environments:

    >>> AutoEnv.list_environments()
    >>> AutoAction.list_actions()
"""

from .auto_env import AutoEnv
from .auto_action import AutoAction

__all__ = [
    "AutoEnv",
    "AutoAction",
]
