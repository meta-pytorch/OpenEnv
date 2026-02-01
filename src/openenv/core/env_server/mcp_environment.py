# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP Environment base class for OpenEnv.

This module provides the MCPEnvironment base class that integrates FastMCP servers
with OpenEnv's Gym-style Environment interface. It handles MCP tool discovery
and invocation through the step() API, following RFC 003.

Key features:
- Automatic routing of ListToolsAction and CallToolAction to MCP server
- Reserved tool name validation (reset, step, state, close are protected)
- Timeout handling for tool calls
- Proper error categorization (tool not found, execution errors, timeouts)

Usage:
    from fastmcp import FastMCP
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    class MyMCPEnv(MCPEnvironment):
        def __init__(self):
            mcp = FastMCP("my-server")

            @mcp.tool()
            def my_tool(arg: str) -> str:
                return f"Result: {arg}"

            super().__init__(mcp)

        def reset(self, seed=None, episode_id=None, **kwargs):
            # Reset logic here
            ...

        def _step_impl(self, action):
            # Handle non-MCP actions
            ...

        @property
        def state(self):
            # Return current state
            ...
"""

import asyncio
from abc import abstractmethod
from typing import Any, Optional

from fastmcp import Client

from ..utils import run_async_safely
from .interfaces import Environment
from .mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    RESERVED_TOOL_NAMES,
    Tool,
    ToolError,
    ToolErrorType,
)
from .types import Action, Observation


# Default timeout for MCP tool calls in seconds
MCP_TOOL_CALL_TIMEOUT = 30.0


class MCPEnvironment(Environment):
    """
    Base class for environments that expose tools via MCP (Model Context Protocol).

    MCPEnvironment bridges FastMCP servers with OpenEnv's Gym-style API, allowing
    agents to discover and invoke MCP tools through the standard step() interface.

    The class automatically handles:
    - ListToolsAction: Returns available tools from the MCP server
    - CallToolAction: Invokes a specific tool with arguments

    All other actions are delegated to the abstract _step_impl() method,
    which subclasses must implement.

    Args:
        mcp_server: A FastMCP server instance containing tool definitions.
            The server's tools will be validated against reserved names.
        transform: Optional transform to apply to observations (inherited from Environment).

    Raises:
        ValueError: If any tool in the MCP server uses a reserved name
            (reset, step, state, close).

    Example:
        >>> from fastmcp import FastMCP
        >>> mcp = FastMCP("calculator")
        >>> @mcp.tool()
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> env = MyMCPEnvironment(mcp)
        >>> obs = env.step(ListToolsAction())
        >>> obs.tools[0].name
        'add'
    """

    def __init__(self, mcp_server: Any, transform: Optional[Any] = None) -> None:
        """
        Initialize the MCP environment.

        Args:
            mcp_server: A FastMCP server instance with tool definitions.
            transform: Optional transform to apply to observations.

        Raises:
            ValueError: If any tool uses a reserved name (reset, step, state, close).
        """
        super().__init__(transform=transform)

        # Validate tool names before storing
        self._validate_tool_names(mcp_server)

        self.mcp_server = mcp_server
        self.mcp_client = Client(mcp_server)

    def _validate_tool_names(self, mcp_server: Any) -> None:
        """
        Validate that no tools use reserved names.

        Reserved names (reset, step, state, close) are protected to maintain
        the dual API boundary between infrastructure and agent APIs.

        Args:
            mcp_server: The FastMCP server to validate.

        Raises:
            ValueError: If any tool uses a reserved name.
        """
        # FastMCP stores tools in _tool_manager._tools dict
        if hasattr(mcp_server, "_tool_manager"):
            tool_manager = mcp_server._tool_manager
            # Check both possible attribute names for tools storage
            tools_dict = None
            if hasattr(tool_manager, "_tools"):
                tools_dict = tool_manager._tools
            elif hasattr(tool_manager, "tools"):
                tools_dict = tool_manager.tools

            if tools_dict:
                tool_names = set(tools_dict.keys())
                conflicts = tool_names & RESERVED_TOOL_NAMES
                if conflicts:
                    raise ValueError(
                        f"MCP tools cannot use reserved names: {sorted(conflicts)}. "
                        f"Reserved names are: {sorted(RESERVED_TOOL_NAMES)}"
                    )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute an action in the environment.

        This method routes MCP-specific actions (ListToolsAction, CallToolAction)
        to the appropriate handlers, while delegating all other actions to
        the subclass's _step_impl() method.

        Args:
            action: The action to execute. Can be:
                - ListToolsAction: Returns available MCP tools
                - CallToolAction: Invokes a specific MCP tool
                - Any other Action: Delegated to _step_impl()
            timeout_s: Optional timeout in seconds for the action.
                Defaults to MCP_TOOL_CALL_TIMEOUT (30s) for MCP actions.
            **kwargs: Additional arguments passed to handlers.

        Returns:
            Observation appropriate to the action type:
                - ListToolsObservation for ListToolsAction
                - CallToolObservation for CallToolAction
                - Subclass-defined Observation for other actions
        """
        if isinstance(action, ListToolsAction):
            return self._handle_list_tools()
        elif isinstance(action, CallToolAction):
            return self._handle_call_tool(action, timeout_s=timeout_s)
        else:
            return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _handle_list_tools(self) -> ListToolsObservation:
        """
        Handle a ListToolsAction by querying the MCP server.

        Returns:
            ListToolsObservation containing all available tools with their
            names, descriptions, and input schemas.
        """
        try:
            # Run the async list_tools call synchronously
            # Use run_async_safely to handle both sync and async contexts
            tools_result = run_async_safely(self._async_list_tools())

            # Convert MCP tool objects to our Tool model
            tools = [
                Tool(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                )
                for tool in tools_result
            ]

            return ListToolsObservation(tools=tools)

        except Exception as e:
            # Return an observation with error in metadata
            return ListToolsObservation(
                tools=[],
                metadata={
                    "error": str(e),
                    "error_type": "list_tools_failed",
                },
            )

    async def _async_list_tools(self) -> list:
        """
        Async helper to list tools from the MCP client.

        Returns:
            List of tool objects from the MCP server.
        """
        async with self.mcp_client:
            return await self.mcp_client.list_tools()

    def _handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: Optional[float] = None,
    ) -> CallToolObservation:
        """
        Handle a CallToolAction by invoking the specified tool.

        Args:
            action: The CallToolAction containing tool_name and arguments.
            timeout_s: Timeout in seconds. Defaults to MCP_TOOL_CALL_TIMEOUT (30s).

        Returns:
            CallToolObservation with the tool's result or an error.
        """
        timeout = timeout_s if timeout_s is not None else MCP_TOOL_CALL_TIMEOUT

        try:
            # Run the async call_tool with timeout
            # Use run_async_safely to handle both sync and async contexts
            result = run_async_safely(
                asyncio.wait_for(
                    self._async_call_tool(action.tool_name, action.arguments),
                    timeout=timeout,
                )
            )

            return CallToolObservation(
                tool_name=action.tool_name,
                result=result,
            )

        except asyncio.TimeoutError:
            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=ToolErrorType.TIMEOUT,
                    message=f"Tool '{action.tool_name}' timed out after {timeout} seconds",
                ),
            )

        except Exception as e:
            error_message = str(e)

            # Determine error type based on the exception
            if (
                "not found" in error_message.lower()
                or "unknown tool" in error_message.lower()
            ):
                error_type = ToolErrorType.TOOL_NOT_FOUND
            elif (
                "invalid" in error_message.lower()
                or "argument" in error_message.lower()
            ):
                error_type = ToolErrorType.INVALID_ARGS
            else:
                error_type = ToolErrorType.EXECUTION_ERROR

            return CallToolObservation(
                tool_name=action.tool_name,
                result=None,
                error=ToolError(
                    error_type=error_type,
                    message=error_message,
                ),
            )

    async def _async_call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Async helper to call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            The result from the tool execution.
        """
        async with self.mcp_client:
            return await self.mcp_client.call_tool(tool_name, arguments)

    @abstractmethod
    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions in the environment.

        Subclasses must implement this method to handle any actions that are
        not ListToolsAction or CallToolAction. This is where environment-specific
        action processing should occur.

        Args:
            action: The action to execute (guaranteed not to be an MCP action).
            timeout_s: Optional timeout in seconds.
            **kwargs: Additional arguments.

        Returns:
            An Observation appropriate for the action.
        """
        pass

    def close(self) -> None:
        """
        Clean up resources used by the environment.

        This method cleans up the MCP client and any other resources.
        Subclasses should call super().close() if they override this method.
        """
        # The MCP client uses async context manager, so cleanup happens
        # automatically when the context exits. We just clear references.
        self.mcp_client = None
        self.mcp_server = None
