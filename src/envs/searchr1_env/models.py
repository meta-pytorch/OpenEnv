# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Searchr1 Env Environment.

The searchr1_env environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class Searchr1Action(Action):
    """Action for the Searchr1 Env environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class Searchr1Observation(Observation):
    """Observation from the Searchr1 Env environment - the echoed message."""

    echoed_message: str
    message_length: int = 0

