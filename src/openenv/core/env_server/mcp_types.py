# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP (Model Context Protocol) type definitions for OpenEnv.

This module defines strongly typed models for MCP tool discovery and invocation,
following RFC 003. These types map MCP's REST-like API (tools/list, tools/call)
to Gym-style action types.

Key design decisions:
- Tool discovery (list_tools) does NOT require reset() first
- Reserved tool names (reset, step, state, close) are prohibited
- Both step() and WebSocket /mcp paths are supported
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types import Action, Observation, BaseMessage


class Tool(BaseModel):
    """
    Strongly typed MCP tool specification.

    Follows the MCP ToolSpec format for tool discovery.
    See: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(
        description="Human-readable description of what the tool does"
    )
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema for the tool's input parameters"
    )


class ToolErrorType(str, Enum):
    """Types of errors that can occur during tool execution."""

    EXECUTION_ERROR = "execution_error"  # Tool ran but failed
    INVALID_ARGS = "invalid_args"  # Invalid arguments provided
    TRANSPORT_ERROR = "transport_error"  # Communication failure
    TOOL_NOT_FOUND = "tool_not_found"  # Tool doesn't exist
    TIMEOUT = "timeout"  # Operation timed out


class ToolError(BaseModel):
    """
    Structured error for tool execution failures.

    This is used for transport/framework errors, NOT for errors returned
    by the tool itself (those go in the result field).
    """

    model_config = ConfigDict(extra="forbid")

    error_type: ToolErrorType = Field(description="Category of the error")
    message: str = Field(description="Human-readable error message")


# --- MCP Actions ---


class ListToolsAction(Action):
    """
    Request list of available tools from the environment.

    This action triggers MCP's tools/list operation and returns
    all available tools with their schemas.

    Note: Does NOT require reset() to be called first.
    """

    type: Literal["list_tools"] = Field(
        default="list_tools", description="Action type discriminator"
    )


class CallToolAction(Action):
    """
    Call a specific tool via MCP.

    This action triggers MCP's tools/call operation with the
    specified tool name and arguments.
    """

    type: Literal["call_tool"] = Field(
        default="call_tool", description="Action type discriminator"
    )
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


# --- MCP Observations ---


class ListToolsObservation(Observation):
    """
    Response containing available tools.

    Returned when processing a ListToolsAction.
    """

    tools: List[Tool] = Field(description="List of available tools with their schemas")


class CallToolObservation(Observation):
    """
    Response from tool execution.

    Contains the tool's result or an error if the call failed.
    Tool-specific errors (from the tool itself) are included in the result.
    Transport/framework errors use the error field.
    """

    tool_name: str = Field(description="Name of the tool that was called")
    result: Any = Field(
        default=None, description="Tool-specific result (may include tool errors)"
    )
    error: Optional[ToolError] = Field(
        default=None, description="Transport/framework error if call failed"
    )


# --- WebSocket Message Types for MCP ---


class WSMCPMessage(BaseMessage):
    """
    WebSocket message for MCP JSON-RPC requests.

    Allows direct MCP access via WebSocket for production inference,
    bypassing the step() API.
    """

    type: Literal["mcp"] = Field(default="mcp", description="Message type")
    data: Dict[str, Any] = Field(description="JSON-RPC payload (method, params, id)")


class WSMCPResponse(BaseModel):
    """
    WebSocket response for MCP JSON-RPC.

    Contains the JSON-RPC response from the MCP server.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="mcp", description="Response type")
    data: Dict[str, Any] = Field(description="JSON-RPC response payload")


# Reserved tool names that cannot be used (protects dual API boundary)
RESERVED_TOOL_NAMES = frozenset(["reset", "step", "state", "close"])
