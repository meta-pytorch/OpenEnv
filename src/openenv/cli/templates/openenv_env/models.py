# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the __ENV_TITLE_NAME__ Environment.

The __ENV_NAME__ environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

from openenv.core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class __ENV_CLASS_NAME__Action(Action):
    """Action for the __ENV_TITLE_NAME__ environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class __ENV_CLASS_NAME__Observation(Observation):
    """Observation from the __ENV_TITLE_NAME__ environment - the echoed message."""

    echoed_message: str
    message_length: int = 0

