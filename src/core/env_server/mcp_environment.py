# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCPEnvironment base class for environments that expose MCP tools.

This module provides a base class that handles all the boilerplate for
MCP-based environments, allowing developers to just define their tools
using FastMCP decorators.
"""

from __future__ import annotations

from typing import Any

from .interfaces import Environment
from .types import Action, CallToolAction, ListToolsAction, Observation, State


class MCPEnvironment(Environment):
    """
    Base class for environments that expose tools via MCP.

    This class handles all the boilerplate of setting up MCP client/server
    communication. Subclasses just need to provide a FastMCP server instance
    with tools registered via decorators.

    The environment automatically handles ListToolsAction and CallToolAction,
    delegating to the configured MCP server.

    IMPORTANT: MCPEnvironment is designed to be used via HTTPEnvServer (FastAPI).
    The step() method should not be called directly due to async/await requirements
    of the MCP client. HTTPEnvServer automatically routes MCP actions to the async
    _handle_mcp_action() method.

    Example:
        >>> from fastmcp import FastMCP
        >>> from core.env_server import MCPEnvironment
        >>>
        >>> # Define MCP server with tools
        >>> mcp = FastMCP("my_env")
        >>>
        >>> @mcp.tool()
        >>> def my_tool(param: str) -> dict:
        >>>     return {"result": param}
        >>>
        >>> # Create environment
        >>> env = MCPEnvironment(mcp)
        >>>
        >>> # Use with HTTPEnvServer
        >>> from core.env_server import create_fastapi_app
        >>> from core.models import Action, Observation
        >>> app = create_fastapi_app(env, Action, Observation)
    """

    def __init__(self, mcp_server: Any):
        """
        Initialize MCP environment.

        Args:
            mcp_server: FastMCP server instance with tools registered
        """
        from fastmcp import Client

        self.mcp_server = mcp_server
        self.mcp_client = Client(mcp_server)
        super().__init__(mcp_client=self.mcp_client)

    def reset(self) -> Observation:
        """
        Reset the environment.

        Returns initial observation with done=False. MCP environments are
        stateless by default (state is managed by the MCP server if needed).
        """
        return Observation(done=False)

    def step(self, action: Action) -> Observation:
        """
        Execute an MCP action in the environment.

        MCPEnvironment ONLY accepts MCP actions (ListToolsAction, CallToolAction).

        NOTE: This is a sync method that internally runs async MCP operations.
        When called from sync context, it uses asyncio.run(). When called from
        HTTPEnvServer (async context), HTTPEnvServer intercepts MCP actions and
        calls _handle_mcp_action() directly to avoid blocking the event loop.

        Args:
            action: MCP action to execute (ListToolsAction or CallToolAction)

        Returns:
            Observation from action execution

        Raises:
            ValueError: If action is not an MCP action type
        """
        import asyncio

        if not isinstance(action, (ListToolsAction, CallToolAction)):
            raise ValueError(
                f"MCP environments only accept MCP actions (ListToolsAction, CallToolAction). "
                f"Got: {type(action).__name__}"
            )

        # Handle MCP action - run async operation in sync context
        return asyncio.run(self._handle_mcp_action(action))

    async def _handle_mcp_action(self, action: Action) -> Observation:
        """
        Handle MCP actions asynchronously.

        This method is called by HTTPEnvServer to handle MCP actions without
        blocking the asyncio event loop.

        Args:
            action: ListToolsAction or CallToolAction

        Returns:
            ListToolsObservation or CallToolObservation

        Raises:
            ValueError: If MCP client not configured or action type invalid
        """
        from .types import CallToolObservation, ListToolsObservation

        if self.mcp_client is None:
            raise ValueError("MCP client not configured for this environment")

        async with self.mcp_client:
            if isinstance(action, ListToolsAction):
                tools = await self.mcp_client.list_tools()
                return ListToolsObservation(
                    done=False,
                    tools=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.inputSchema,
                        }
                        for tool in tools
                    ],
                )

            elif isinstance(action, CallToolAction):
                try:
                    result = await self.mcp_client.call_tool(
                        action.tool_name, action.parameters
                    )
                    # Extract data from CallToolResult (FastMCP wraps results)
                    result_data = result.data if hasattr(result, "data") else result
                    return CallToolObservation(
                        done=False, result=result_data, tool_name=action.tool_name
                    )
                except Exception as e:
                    return CallToolObservation(
                        done=False, error=str(e), tool_name=action.tool_name
                    )

            else:
                raise ValueError(f"Unsupported MCP action type: {type(action)}")

    @property
    def state(self) -> State:
        """
        Get current environment state.

        MCP environments are stateless by default. State management can be
        implemented in the MCP server using FastMCP's session persistence.
        """
        return State()
