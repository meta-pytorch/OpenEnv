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
from enum import Enum
from typing import Any, Dict, List, Optional

from .types import Action, Observation


class ToolErrorType(Enum):
    """Types of errors that can occur during tool execution."""

    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_NOT_FOUND = "tool_not_found"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


@dataclass
class ToolError:
    """
    Structured error for tool call failures.

    Used for transport/validation errors. Tool execution errors that are
    part of normal operation should be returned in the result field.
    """

    error_type: ToolErrorType
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    """
    Strongly typed representation of an MCP tool.

    Follows the MCP specification for tool definitions.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


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

    tools: List[Tool] = field(default_factory=list)


@dataclass(kw_only=True)
class CallToolObservation(Observation):
    """
    Observation returned from CallToolAction.

    Contains the result of calling a tool. The error field is for
    transport/validation errors only - tool execution errors should
    be part of the result.
    """

    tool_name: str
    result: Any = None
    error: Optional[ToolError] = None
