# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the TB2 environment.
"""

from pydantic import Field


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation, State


class Tbench2Action(Action):
    """Action for interacting with a TB2 task session."""

    action_type: str = Field(default="exec")
    command: str = Field(default="")
    session_id: str | None = Field(default=None)
    block: bool = Field(default=True)
    wait_seconds: float | None = Field(default=None)
    file_path: str = Field(default="")
    content: str = Field(default="")


class Tbench2Observation(Observation):
    """Observation returned from the TB2 environment."""

    instruction: str = Field(default="")
    output: str = Field(default="")
    success: bool = Field(default=True)
    error: str = Field(default="")
    task_id: str = Field(default="")
    task_path: str = Field(default="")
    session_id: str | None = Field(default=None)
    action_type: str = Field(default="")
    info: dict = Field(default_factory=dict)


class Tbench2State(State):
    """Server-side state for a TB2 task."""

    task_id: str = Field(default="")
    task_path: str = Field(default="")
    session_id: str = Field(default="")
    terminal_ready: bool = Field(default=False)
    last_action_type: str = Field(default="")
    last_command: str = Field(default="")
    last_output: str = Field(default="")
