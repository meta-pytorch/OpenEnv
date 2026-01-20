# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.

Supports both traditional EchoAction and MCP actions (ListToolsAction, CallToolAction)
for backwards compatibility while enabling MCP tool discovery and invocation.
"""

from typing import Union
from uuid import uuid4

from fastmcp import Client

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from openenv.core.env_server.mcp_types import (
        ListToolsAction,
        CallToolAction,
        ListToolsObservation,
        CallToolObservation,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from ..models import EchoAction, EchoObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
    from openenv.core.env_server.mcp_types import (
        ListToolsAction,
        CallToolAction,
        ListToolsObservation,
        CallToolObservation,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from models import EchoAction, EchoObservation

from .mcp_server import mcp


class EchoEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = EchoEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Echo environment ready!"
        >>>
        >>> obs = env.step(EchoAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    def __init__(self):
        """Initialize the echo environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        # Initialize MCP client for the local MCP server
        # Exposed as both private and public for WebSocket MCP support
        self._mcp_client = Client(mcp)
        self.mcp_client = self._mcp_client

    def reset(self) -> EchoObservation:
        """
        Reset the environment.

        Returns:
            EchoObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return EchoObservation(
            echoed_message="Echo environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(
        self, action: Union[EchoAction, ListToolsAction, CallToolAction]
    ) -> Union[EchoObservation, ListToolsObservation, CallToolObservation]:  # type: ignore[override]
        """
        Execute a step in the environment.

        Supports both traditional EchoAction and MCP actions (ListToolsAction, CallToolAction)
        for backwards compatibility while enabling MCP tool discovery and invocation.

        Args:
            action: EchoAction, ListToolsAction, or CallToolAction

        Returns:
            EchoObservation, ListToolsObservation, or CallToolObservation depending on action type
        """
        self._state.step_count += 1

        # Handle MCP ListToolsAction
        if isinstance(action, ListToolsAction):
            return self._handle_list_tools()

        # Handle MCP CallToolAction
        if isinstance(action, CallToolAction):
            return self._handle_call_tool(action)

        # Handle traditional EchoAction (backwards compatibility)
        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return EchoObservation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    def _handle_list_tools(self) -> ListToolsObservation:
        """
        Handle ListToolsAction by returning available MCP tools.

        Returns:
            ListToolsObservation with available tools
        """
        import asyncio

        async def _list_tools():
            async with self._mcp_client:
                return await self._mcp_client.list_tools()

        # Run async operation synchronously using asyncio.run()
        tools_result = asyncio.run(_list_tools())

        # Convert to Tool objects
        tools = [
            Tool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
            )
            for tool in tools_result
        ]

        return ListToolsObservation(
            tools=tools,
            done=False,
            reward=0.0,
        )

    def _handle_call_tool(self, action: CallToolAction) -> CallToolObservation:
        """
        Handle CallToolAction by invoking the specified MCP tool.

        Args:
            action: CallToolAction with tool_name and arguments

        Returns:
            CallToolObservation with tool result or error
        """
        import asyncio

        async def _call_tool():
            async with self._mcp_client:
                return await self._mcp_client.call_tool(
                    action.tool_name, action.arguments
                )

        try:
            # Run async operation synchronously using asyncio.run()
            result = asyncio.run(_call_tool())

            # Extract result content
            if result.content:
                # Get the first content item's text or data
                content = result.content[0]
                if hasattr(content, "text"):
                    tool_result = content.text
                else:
                    tool_result = str(content)
            else:
                tool_result = None

            return CallToolObservation(
                tool_name=action.tool_name,
                result=tool_result,
                error=None,
                done=False,
                reward=0.0,
            )
        except Exception as e:
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=ToolErrorType.EXECUTION_ERROR,
                    message=str(e),
                ),
                done=False,
                reward=0.0,
            )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
