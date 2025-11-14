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
class ToolCallAction(Action):
    """Action representing a named operation with typed parameters.

    Inspired by MCP's tool-calling pattern, but generalized to represent
    ANY environment action - not just tool calls or code execution.

    Examples:
    - Tool calls: tool_name="search_web", parameters={"query": "..."}
    - Code execution: tool_name="execute_code", parameters={"code": "..."}
    - Game moves: tool_name="move_piece", parameters={"from": "e2", "to": "e4"}
    - Navigation: tool_name="go_north", parameters={}
    - Configuration: tool_name="set_timeout", parameters={"seconds": 30}

    This is the standard action type for all OpenEnv environments.
    Environments dispatch based on tool_name to handle different action types.
    """

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""

    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # JSON Schema type: "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Specification of an action that can be taken in an environment.

    Inspired by MCP's tool definition format and compatible with LLM tool-calling
    APIs (Claude, OpenAI, etc.), but represents ANY action type - not just tools.

    This can describe:
    - External tool calls (search_web, read_file)
    - Code execution (execute_python, run_bash)
    - Game actions (move_piece, attack, defend)
    - Navigation commands (go_north, turn_left)
    - Configuration changes (set_parameter, update_config)
    - Any domain-specific action
    """

    name: str
    description: str
    parameters: List["ToolParameter"]

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    p.name: {
                        "type": p.type,
                        "description": p.description,
                    }
                    for p in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }
