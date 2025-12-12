# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for FleetEnvClient (RFC 003 tool-call actions)."""

from dataclasses import dataclass, field
from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    from core.env_server.types import Action
except ImportError:
    from openenv_core.env_server.types import Action


@dataclass(kw_only=True)
class ListToolsAction(Action):
    """Request list of available MCP tools from the Fleet environment."""


@dataclass(kw_only=True)
class CallToolAction(Action):
    """Call a specific MCP tool exposed by the Fleet environment."""

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


