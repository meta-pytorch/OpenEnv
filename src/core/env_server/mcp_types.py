# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# Type aliases
Scalar = Union[int, float, bool]


class ToolErrorType(Enum):
    """Types of errors that can occur during tool execution."""

    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_NOT_FOUND = "tool_not_found"
    TRANSPORT_ERROR = "transport_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


@dataclass
class ToolError:
    """
    Structured error information for tool call failures.

    This captures errors at the infrastructure/transport level, not errors
    that are part of the tool's normal result (those should be in the result field).
    """

    error_type: ToolErrorType
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class Tool:
    """
    Strongly typed representation of an MCP tool.

    Follows the MCP specification for tool definitions with JSON Schema
    for input/output validation.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions."""

    metadata: Dict[str, Any] = field(default_factory=dict)


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
class Observation:
    """Base class for all environment observations."""

    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ListToolsObservation(Observation):
    """
    Observation returned from ListToolsAction.

    Contains the list of available tools with their schemas, following
    the MCP specification format.
    """

    tools: List[Tool] = field(default_factory=list)


@dataclass(kw_only=True)
class CallToolObservation(Observation):
    """
    Observation returned from CallToolAction.

    Contains the result of calling a tool. The result field contains the tool's
    output (including any tool-level errors). The error field is used only for
    infrastructure-level errors (invalid args, transport issues, etc.).
    """

    tool_name: str
    result: Any
    error: Optional[ToolError] = None


@dataclass
class State:
    """Base class for environment state."""

    episode_id: Optional[str] = None
    step_count: int = 0


@dataclass
class CodeExecResult:
    """Result of code execution containing stdout, stderr, and exit code."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass
class EnvironmentMetadata:
    """Metadata about an environment for documentation and UI purposes."""
    
    name: str
    description: str
    readme_content: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    documentation_url: Optional[str] = None
