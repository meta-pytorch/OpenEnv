# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP Client classes for tool-calling environments.

This module provides client classes for interacting with MCP-enabled environments:
- MCPClientBase: Base class with shared tool discovery
- MCPToolClient: Client for tool-calling style (one tool per step)
- MCPHttpClient: Lightweight client for direct HTTP MCP access (RFC 003)

These clients abstract away the MCP protocol details, providing a clean interface
for listing and calling tools on remote environments.

Example using MCPToolClient (WebSocket-based, maintains session):
    >>> from openenv.core.mcp_client import MCPToolClient
    >>>
    >>> with MCPToolClient(base_url="http://localhost:8000") as env:
    ...     # Discover available tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Call a tool
    ...     result = env.call_tool("echo_message", message="Hello!")
    ...     print(result)

Example using MCPHttpClient (direct HTTP, stateless):
    >>> from openenv.core.mcp_client import MCPHttpClient
    >>>
    >>> client = MCPHttpClient(base_url="http://localhost:8000")
    >>> tools = client.list_tools()
    >>> result = client.call_tool("echo_message", message="Hello!")
"""

import httpx
from typing import Any, Dict, List, Optional

from .client_types import StepResult, StateT
from .env_client import EnvClient
from .env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    MCPRequest,
    MCPResponse,
    Tool,
    ToolError,
)
from .env_server.types import Observation, State


class MCPClientBase(EnvClient[Any, Observation, State]):
    """
    Base class for MCP clients with tool discovery.

    This class provides the common `list_tools()` method for discovering
    available tools from an MCP-enabled environment. Subclasses implement
    specific interaction patterns (tool-calling or CodeAct).

    Attributes:
        _tools_cache: Cached list of tools (populated on first `list_tools()` call)
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Optional[Any] = None,
    ):
        """
        Initialize MCP client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
            connect_timeout_s: Timeout for establishing WebSocket connection.
            message_timeout_s: Timeout for receiving responses to messages.
            provider: Optional container/runtime provider for lifecycle management.
        """
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )
        self._tools_cache: Optional[List[Tool]] = None

    def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """
        Discover available tools from the environment.

        Args:
            use_cache: If True, return cached tools if available.
                      Set to False to force a fresh request.

        Returns:
            List of Tool objects with name, description, and input_schema.

        Example:
            >>> tools = env.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        result = self.step(ListToolsAction())
        self._tools_cache = result.observation.tools
        return self._tools_cache

    def _step_payload(self, action: Any) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        if isinstance(action, ListToolsAction):
            return {"type": "list_tools"}
        elif isinstance(action, CallToolAction):
            return {
                "type": "call_tool",
                "tool_name": action.tool_name,
                "arguments": action.arguments,
            }
        else:
            # For unknown actions, try to serialize as dict
            if hasattr(action, "model_dump"):
                return action.model_dump()
            return {"action": str(action)}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        """Convert a JSON response from the env server to StepResult[Observation]."""
        obs_data = payload.get("observation", {})

        # Check if this is a ListToolsObservation
        if "tools" in obs_data:
            tools = [
                Tool(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    input_schema=t.get("input_schema", t.get("inputSchema", {})),
                )
                for t in obs_data.get("tools", [])
            ]
            observation = ListToolsObservation(
                tools=tools,
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        # Check if this is a CallToolObservation
        elif "tool_name" in obs_data:
            error = None
            if obs_data.get("error"):
                error = ToolError(**obs_data["error"])

            observation = CallToolObservation(
                tool_name=obs_data.get("tool_name", ""),
                result=obs_data.get("result"),
                error=error,
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        else:
            # Generic observation
            observation = Observation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Convert a JSON response from the state endpoint to a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


class MCPToolClient(MCPClientBase):
    """
    Client for tool-calling style MCP interactions.

    Each step invokes a single tool. Use this for traditional function-calling
    agent patterns where the agent decides which tool to call next.

    This client provides convenience methods for tool discovery and invocation:
    - `list_tools()`: Get all available tools with their schemas
    - `call_tool(name, **kwargs)`: Invoke a tool by name with arguments

    Example:
        >>> with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     # Reset the environment
        ...     env.reset()
        ...
        ...     # Discover available tools
        ...     tools = env.list_tools()
        ...     print([t.name for t in tools])  # ['echo_message', 'echo_with_length']
        ...
        ...     # Call a tool directly
        ...     result = env.call_tool("echo_message", message="Hello!")
        ...     print(result)  # "Hello!"
        ...
        ...     # Or use the full action interface
        ...     from openenv.core.env_server.mcp_types import CallToolAction
        ...     step_result = env.step(CallToolAction(
        ...         tool_name="echo_with_length",
        ...         arguments={"message": "Test"}
        ...     ))
        ...     print(step_result.observation.result)
    """

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """
        Call a tool by name.

        This is a convenience method that creates a CallToolAction, executes it,
        and returns the result directly. For more control, use `step()` with
        a CallToolAction directly.

        Args:
            name: Name of the tool to invoke (must match a tool from `list_tools()`).
            **kwargs: Arguments to pass to the tool. Must match the tool's input_schema.

        Returns:
            The tool's result. The type depends on the tool being called.

        Raises:
            RuntimeError: If the server returns an error response.

        Example:
            >>> result = env.call_tool("add", a=5, b=3)
            >>> print(result)  # 8
            >>>
            >>> result = env.call_tool("greet", name="Claude")
            >>> print(result)  # "Hello, Claude!"
        """
        action = CallToolAction(tool_name=name, arguments=kwargs)
        result = self.step(action)
        obs = result.observation

        # Check for transport/framework errors
        if isinstance(obs, CallToolObservation) and obs.error is not None:
            raise RuntimeError(
                f"Tool '{name}' failed: {obs.error.message} "
                f"(type: {obs.error.error_type.value})"
            )

        # Return the result
        if isinstance(obs, CallToolObservation):
            result = obs.result
            # Handle FastMCP CallToolResult objects
            # - As object: has .data attribute
            # - As dict (from JSON): has "data" key
            if hasattr(result, "data"):
                return result.data
            if isinstance(result, dict) and "data" in result:
                return result["data"]
            return result

        # Fallback for unexpected observation types
        return obs

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.

        Args:
            name: Name of the tool to find.

        Returns:
            The Tool object if found, None otherwise.

        Example:
            >>> tool = env.get_tool("echo_message")
            >>> if tool:
            ...     print(tool.description)
            ...     print(tool.input_schema)
        """
        tools = self.list_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists.

        Args:
            name: Name of the tool to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        return self.get_tool(name) is not None


class MCPHttpClient:
    """
    Lightweight HTTP client for direct MCP access (RFC 003).

    This client uses the POST /mcp endpoint for stateless tool discovery and
    invocation. Unlike MCPToolClient, it doesn't maintain a WebSocket session,
    making it ideal for:
    - Production/inference scenarios
    - Simple tool calls without episode state
    - Integration with existing HTTP-based infrastructure

    Note: This client doesn't support reset/step/state operations. Use
    MCPToolClient for full environment interaction with session state.

    Example:
        >>> client = MCPHttpClient(base_url="http://localhost:8000")
        >>>
        >>> # List available tools
        >>> tools = client.list_tools()
        >>> for tool in tools:
        ...     print(f"{tool.name}: {tool.description}")
        >>>
        >>> # Call a tool
        >>> result = client.call_tool("echo_message", message="Hello!")
        >>> print(result)
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 30.0,
    ):
        """
        Initialize MCP HTTP client.

        Args:
            base_url: Base URL of the environment server (http:// only).
            timeout_s: Timeout for HTTP requests in seconds.
        """
        # Normalize base_url
        self.base_url = base_url.rstrip("/")
        if self.base_url.startswith("ws://"):
            self.base_url = self.base_url.replace("ws://", "http://", 1)
        elif self.base_url.startswith("wss://"):
            self.base_url = self.base_url.replace("wss://", "https://", 1)

        self.timeout_s = timeout_s
        self._tools_cache: Optional[List[Tool]] = None
        self._client = httpx.Client(timeout=timeout_s)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the HTTP client."""
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def _mcp_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send an MCP JSON-RPC request to the /mcp endpoint.

        Args:
            method: MCP method name (e.g., "tools/list", "tools/call")
            params: Optional method parameters

        Returns:
            The 'result' field from the JSON-RPC response

        Raises:
            RuntimeError: If the server returns an error response
        """
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1,
        }
        if params:
            request_data["params"] = params

        response = self._client.post(
            f"{self.base_url}/mcp",
            json=request_data,
        )
        response.raise_for_status()

        data = response.json()

        if "error" in data and data["error"]:
            error = data["error"]
            raise RuntimeError(
                f"MCP error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}"
            )

        return data.get("result")

    def list_tools(self, use_cache: bool = True) -> List[Tool]:
        """
        Discover available tools from the environment.

        Args:
            use_cache: If True, return cached tools if available.
                      Set to False to force a fresh request.

        Returns:
            List of Tool objects with name, description, and input_schema.

        Example:
            >>> tools = client.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        result = self._mcp_request("tools/list")
        tools_data = result.get("tools", [])

        self._tools_cache = [
            Tool(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", t.get("input_schema", {})),
            )
            for t in tools_data
        ]

        return self._tools_cache

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """
        Call a tool by name.

        Args:
            name: Name of the tool to invoke.
            **kwargs: Arguments to pass to the tool.

        Returns:
            The tool's result. The type depends on the tool being called.

        Raises:
            RuntimeError: If the server returns an error response.

        Example:
            >>> result = client.call_tool("add", a=5, b=3)
            >>> print(result)  # 8
            >>>
            >>> result = client.call_tool("greet", name="Claude")
            >>> print(result)  # "Hello, Claude!"
        """
        result = self._mcp_request(
            "tools/call",
            params={"name": name, "arguments": kwargs},
        )

        # Handle FastMCP CallToolResult objects
        if isinstance(result, dict):
            # Check for nested data structure
            if "data" in result:
                return result["data"]
            # Check for content array (MCP standard format)
            if "content" in result and isinstance(result["content"], list):
                contents = result["content"]
                if len(contents) == 1:
                    content = contents[0]
                    if isinstance(content, dict) and "text" in content:
                        return content["text"]
                return contents

        return result

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a specific tool by name.

        Args:
            name: Name of the tool to find.

        Returns:
            The Tool object if found, None otherwise.
        """
        tools = self.list_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists.

        Args:
            name: Name of the tool to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        return self.get_tool(name) is not None
