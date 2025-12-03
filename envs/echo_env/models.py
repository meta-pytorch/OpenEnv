# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Echo Environment.

The Echo environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class EchoAction(Action):
    """Action for the Echo environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class EchoObservation(Observation):
    """Observation from the Echo environment - the echoed message."""

    echoed_message: str
    message_length: int = 0