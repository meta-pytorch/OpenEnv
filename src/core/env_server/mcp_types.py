# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MCP-specific action and observation types.

This module defines types specific to MCP (Model Context Protocol) integration.
Separated from types.py per pankit's suggestion to avoid conflicts with
Python's built-in types module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .types import Action, Observation


@dataclass(kw_only=True)
class ListToolsAction(Action):
    """
    Action to request available tools from MCP servers.

    This action triggers a tools/list call to all configured MCP servers,
    returning their tool schemas in the observation.
    """

    pass


@dataclass(kw_only=True)
class CallToolAction(Action):
    """
    Action to call a specific tool via MCP.

    Triggers a tools/call request to the appropriate MCP server.
    """

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ListToolsObservation(Observation):
    """
    Observation returned from ListToolsAction.

    Contains the list of available tools with their schemas.
    """

    tools: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(kw_only=True)
class CallToolObservation(Observation):
    """
    Observation returned from CallToolAction.

    Contains the result of calling a tool, or an error if the call failed.
    """

    result: Optional[Any] = None
    error: Optional[str] = None
    tool_name: Optional[str] = None
