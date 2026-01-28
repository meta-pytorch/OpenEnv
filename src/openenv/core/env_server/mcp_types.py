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
- Both step(), WebSocket /mcp, and HTTP POST /mcp paths are supported
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

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


def _get_action_type(v: Any) -> str:
    """Get the action type discriminator for MCP actions."""
    if isinstance(v, dict):
        return v.get("type", "call_tool")
    return getattr(v, "type", "call_tool")


# Union type for polymorphic MCP action deserialization
MCPAction = Annotated[
    Union[
        Annotated[ListToolsAction, Tag("list_tools")],
        Annotated[CallToolAction, Tag("call_tool")],
    ],
    Discriminator(_get_action_type),
]
"""
Union type for MCP actions with automatic type discrimination.

This allows both ListToolsAction and CallToolAction to be deserialized
from the same endpoint based on the 'type' field:
- {"type": "list_tools"} -> ListToolsAction
- {"type": "call_tool", "tool_name": "...", "arguments": {...}} -> CallToolAction

Usage:
    from pydantic import TypeAdapter
    adapter = TypeAdapter(MCPAction)
    action = adapter.validate_python({"type": "list_tools"})
"""


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


# --- HTTP MCP Types (RFC 003) ---


class MCPRequest(BaseModel):
    """
    HTTP request for MCP JSON-RPC.

    Supports the MCP protocol via HTTP POST /mcp endpoint for production/inference
    use cases, bypassing the step() API.

    See RFC 003: MCP Support for details.

    Example requests:
        # List tools
        {"jsonrpc": "2.0", "method": "tools/list", "id": 1}

        # Call a tool
        {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "search", "arguments": {"query": "test"}}, "id": 2}
    """

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = Field(
        default="2.0", description="JSON-RPC version (must be 2.0)"
    )
    method: str = Field(description="MCP method to call (tools/list, tools/call)")
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Method parameters (required for tools/call)"
    )
    id: Optional[Any] = Field(
        default=None, description="Request ID for matching responses"
    )


class MCPResponse(BaseModel):
    """
    HTTP response for MCP JSON-RPC.

    Contains the JSON-RPC response from the MCP server.

    See RFC 003: MCP Support for details.

    Example responses:
        # Success
        {"jsonrpc": "2.0", "result": {"tools": [...]}, "id": 1}

        # Error
        {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": 1}
    """

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"] = Field(
        default="2.0", description="JSON-RPC version (always 2.0)"
    )
    result: Optional[Any] = Field(default=None, description="Method result on success")
    error: Optional[Dict[str, Any]] = Field(
        default=None, description="Error object on failure"
    )
    id: Optional[Any] = Field(default=None, description="Request ID (matches request)")


# Reserved tool names that cannot be used (protects dual API boundary)
RESERVED_TOOL_NAMES = frozenset(["reset", "step", "state", "close"])
