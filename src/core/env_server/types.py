# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Type aliases
Scalar = Union[int, float, bool]


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
