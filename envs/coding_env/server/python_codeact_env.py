# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python Code Action Environment with MCP Support.

This module provides a server-side environment implementation for executing
Python code actions using PyExecutor. It supports both:
- MCP tool-calling style (via ListToolsAction/CallToolAction)
- Legacy CodeAction for backward compatibility

MCP Tools exposed:
- `execute_code(code)`: Execute Python code and return results
- `check_syntax(code)`: Check Python code syntax without executing
"""

import ast
import uuid
from typing import Any, Optional

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation
from .python_executor import PyExecutor

from ..models import CodeAction, CodeObservation, CodeState
from .transforms import create_safe_coding_transform


class PythonCodeActEnv(MCPEnvironment):
    """
    Python Code Action Environment for executing code and tracking state.

    This environment executes Python code and exposes functionality through MCP tools:
    - `execute_code(code)`: Execute Python code and return stdout/stderr/exit_code
    - `check_syntax(code)`: Check Python code syntax without executing

    It also maintains backward compatibility with the legacy CodeAction for
    existing clients.

    Example using MCP tools:
        >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
        >>> env = PythonCodeActEnv()
        >>> env.reset()
        >>>
        >>> # List available tools
        >>> obs = env.step(ListToolsAction())
        >>> print([t.name for t in obs.tools])  # ["execute_code", "check_syntax"]
        >>>
        >>> # Execute code via MCP tool
        >>> obs = env.step(CallToolAction(
        ...     tool_name="execute_code",
        ...     arguments={"code": "print('Hello, World!')"}
        ... ))
        >>> print(obs.result)  # {"stdout": "Hello, World!\\n", "stderr": "", "exit_code": 0}

    Example using legacy CodeAction (backward compatible):
        >>> env = PythonCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(code="print('Hello, World!')")
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, World!\\n"
    """

    def __init__(self):
        """Initialize the coding environment with MCP server and tools."""
        # Create executor first (needed by tools)
        self._executor = PyExecutor()
        self._state = CodeState()
        self.transform = create_safe_coding_transform()

        # Create MCP server and define tools
        mcp = FastMCP("coding_env")

        # Store reference to self for tool closures
        env_self = self

        @mcp.tool
        def execute_code(code: str) -> dict:
            """
            Execute Python code and return results.

            The execution context is persistent within an episode - variables and
            functions defined in previous steps are available in subsequent steps.

            Args:
                code: Python code to execute

            Returns:
                Dictionary with:
                - stdout: Standard output from the code execution
                - stderr: Standard error from the code execution
                - exit_code: 0 for success, non-zero for errors
            """
            result = env_self._executor.run(code)

            # Update state
            env_self._state.step_count += 1
            env_self._state.last_exit_code = result.exit_code

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }

        @mcp.tool
        def check_syntax(code: str) -> dict:
            """
            Check Python code syntax without executing.

            Args:
                code: Python code to check

            Returns:
                Dictionary with:
                - valid: True if syntax is valid, False otherwise
                - error: Error message if syntax is invalid, None otherwise
                - line: Line number of error if applicable
            """
            try:
                ast.parse(code)
                return {"valid": True, "error": None, "line": None}
            except SyntaxError as e:
                return {
                    "valid": False,
                    "error": str(e.msg) if e.msg else str(e),
                    "line": e.lineno,
                }

        # Pass the MCP server to the base class
        super().__init__(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset environment and start fresh execution session.

        Args:
            seed: Optional random seed (unused)
            episode_id: Optional episode ID to use

        Returns:
            Initial observation with empty stdout/stderr and exit_code=0
        """
        # Initialize fresh state
        self._state = CodeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self._state.last_exit_code = 0

        # Reset executor to clear any previously defined variables/functions
        self._executor = PyExecutor()

        # Reset transform to clear any accumulated state
        self.transform = create_safe_coding_transform()

        # Return initial observation
        observation = CodeObservation(
            stdout="",
            stderr="",
            exit_code=0,
            metadata={"status": "ready"},
        )

        return self._apply_transform(observation)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions (legacy CodeAction support).

        This method provides backward compatibility with the legacy CodeAction
        interface. New clients should use MCP tools instead.

        Args:
            action: CodeAction containing the code to execute

        Returns:
            CodeObservation with execution results (stdout, stderr, exit_code)
        """
        if isinstance(action, CodeAction):
            # Execute the code using PyExecutor
            result = self._executor.run(action.code)

            # Update state
            self._state.step_count += 1
            self._state.last_exit_code = result.exit_code

            # Create observation from execution result
            observation = CodeObservation(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                metadata={"last_code": action.code},
            )

            return self._apply_transform(observation)

        # Unknown action type
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction, CallToolAction, or CodeAction."
            },
        )

    @property
    def state(self) -> CodeState:
        """Get current environment state including last exit code."""
        return self._state
