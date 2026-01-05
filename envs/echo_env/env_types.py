# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Echo Environment.

The Echo environment is a simple test environment that echoes back messages.
"""

from pydantic import Field

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation


class EchoAction(Action):
    """Action for the Echo environment - just a message to echo."""

    message: str = Field(..., min_length=1, description="Message to echo back")


class EchoObservation(Observation):
    """Observation from the Echo environment - the echoed message."""

    echoed_message: str = Field(..., description="The echoed message from the environment")
    message_length: int = Field(default=0, ge=0, description="Length of the echoed message")