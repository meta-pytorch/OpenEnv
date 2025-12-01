# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HTTP server wrapper for Environment instances.

This module provides utilities to wrap any Environment subclass and expose it
over HTTP endpoints that HTTPEnvClient can consume.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, Optional, Type

from fastapi import Body, FastAPI, Request

from .interfaces import Environment
from .mcp_environment import MCPEnvironment
from .mcp_types import Action, CallToolAction, ListToolsAction, Observation


class HTTPEnvServer:
    """
    HTTP server wrapper for Environment instances.

    This class wraps an Environment and exposes its reset(), step(), and state
    methods as HTTP endpoints compatible with HTTPEnvClient.

    The server expects:
    - Action deserialization: Converts JSON dict to Action subclass
    - Observation serialization: Converts Observation subclass to JSON dict

    Example:
        >>> from core.env_server import HTTPEnvServer
        >>> from envs.coding_env.server import CodeExecutionEnvironment
        >>>
        >>> env = CodeExecutionEnvironment()
        >>> server = HTTPEnvServer(env)
        >>>
        >>> # Register routes with FastAPI
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> server.register_routes(app)
    """

    def __init__(
        self,
        env: Environment,
        action_cls: Type[Action],
        observation_cls: Type[Observation],
    ):
        """
        Initialize HTTP server wrapper.

        Args:
            env: The Environment instance to wrap
            action_cls: The Action subclass this environment expects
            observation_cls: The Observation subclass this environment returns
        """
        self.env = env
        self.action_cls = action_cls
        self.observation_cls = observation_cls

        # Create thread pool for running sync code in async context
        # This is needed for environments using sync libraries (e.g., Playwright sync API)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def register_routes(self, app: Any) -> None:
        """
        Register HTTP routes on a FastAPI application.

        Args:
            app: FastAPI application instance
        """

        if not isinstance(app, FastAPI):
            raise TypeError("app must be a FastAPI instance")

        @app.post("/reset")
        async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
            """Reset endpoint - returns initial observation."""
            # TODO: Handle seed, episode_id from request if provided
            # Run sync environment code in thread pool to avoid blocking asyncio loop
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(self._executor, self.env.reset)
            return self._serialize_observation(observation)

        @app.post("/step")
        async def step(request: Dict[str, Any]) -> Dict[str, Any]:
            """Step endpoint - executes action and returns observation."""
            # Support both {"action": {...}} and direct action fields
            action_data = request.get("action", request)
            # TODO: Handle timeout_s, request_id, episode_id from request if provided

            # Deserialize action (handle MCP actions specially)
            action = self._deserialize_action(action_data)

            # Handle MCP actions asynchronously (don't use thread pool for async operations)
            if isinstance(action, (ListToolsAction, CallToolAction)):
                if not isinstance(self.env, MCPEnvironment):
                    raise RuntimeError(
                        f"Environment {type(self.env).__name__} received MCP action "
                        f"but is not a MCP environment."
                    )
                observation = await self.env._handle_mcp_action(action)
            else:
                # Execute regular step in thread pool to avoid blocking asyncio loop
                loop = asyncio.get_event_loop()
                observation = await loop.run_in_executor(
                    self._executor, self.env.step, action
                )

            # Return serialized observation
            return self._serialize_observation(observation)

        @app.get("/state")
        async def get_state() -> Dict[str, Any]:
            """State endpoint - returns current environment state."""
            state = self.env.state
            return asdict(state)

        @app.get("/health")
        async def health() -> Dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.post("/mcp")
        async def mcp_endpoint(request: Request) -> Dict[str, Any]:
            """
            MCP JSON-RPC endpoint for direct tool access (production/inference).

            This endpoint provides direct access to the MCP server without going
            through the step() wrapper. Used for production agents that want to
            call tools directly.

            Accepts JSON-RPC 2.0 requests:
            - method: "tools/list" or "tools/call"
            - params: method-specific parameters
            - id: request ID (echoed in response)

            Returns:
                JSON-RPC 2.0 response
            """
            if not hasattr(self.env, "mcp_client") or self.env.mcp_client is None:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "MCP server not configured for this environment",
                    },
                    "id": None,
                }

            try:
                body = await request.json()
            except (ValueError, TypeError):
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error: Invalid JSON",
                    },
                    "id": None,
                }

            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")

            try:
                # Reuse MCP client from environment (avoids creating duplicate client)
                async with self.env.mcp_client:
                    if method == "tools/list":
                        tools = await self.env.mcp_client.list_tools()
                        return {
                            "jsonrpc": "2.0",
                            "result": {
                                "tools": [
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "inputSchema": tool.inputSchema,
                                    }
                                    for tool in tools
                                ]
                            },
                            "id": request_id,
                        }

                    elif method == "tools/call":
                        tool_name = params.get("name")
                        arguments = params.get("arguments", {})
                        result = await self.env.mcp_client.call_tool(
                            tool_name, arguments
                        )

                        # Extract data from CallToolResult (FastMCP wraps results)
                        result_data = result.data if hasattr(result, "data") else result

                        return {
                            "jsonrpc": "2.0",
                            "result": result_data,
                            "id": request_id,
                        }

                    else:
                        return {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}",
                            },
                            "id": request_id,
                        }

            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}",
                    },
                    "id": request_id,
                }

    def _deserialize_action(self, action_data: Dict[str, Any]) -> Action:
        """
        Convert JSON dict to Action instance.

        Args:
            action_data: Dictionary containing action data

        Returns:
            Action instance

        Note:
            This handles both environment-specific actions and MCP actions
            (ListToolsAction, CallToolAction).
        """
        # Check if this is an MCP action by looking at the action type
        action_type = action_data.get("type") or action_data.get("action_type")

        if action_type == "ListToolsAction":
            return ListToolsAction()

        elif action_type == "CallToolAction":
            tool_name = action_data.get("tool_name")
            if tool_name is None:
                raise ValueError("Missing required field 'tool_name' for CallToolAction")
            return CallToolAction(
                tool_name=tool_name,
                parameters=action_data.get("parameters", {}),
            )

        # Otherwise, use the environment-specific action class
        # Get metadata if present (don't mutate input dict)
        metadata = action_data.get("metadata", {})
        action = self.action_cls(
            **{k: v for k, v in action_data.items() if k != "metadata"}
        )
        action.metadata = metadata
        return action

    def _serialize_observation(self, observation: Observation) -> Dict[str, Any]:
        """
        Convert Observation instance to JSON-compatible dict.

        Args:
            observation: Observation instance

        Returns:
            Dictionary compatible with HTTPEnvClient._parse_result()

        The format matches what HTTPEnvClient expects:
        {
            "observation": {...},  # Observation fields
            "reward": float | None,
            "done": bool,
        }
        """
        obs_dict = asdict(observation)

        # Convert numpy arrays to lists for JSON serialization
        def _convert_numpy(obj):
            """Recursively convert numpy arrays to lists."""
            if hasattr(obj, "__array__"):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_convert_numpy(item) for item in obj)
            return obj

        obs_dict = _convert_numpy(obs_dict)

        # Extract reward and done (these are part of StepResult on client side)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)

        # Return in HTTPEnvClient expected format
        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
        }


def create_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
) -> Any:
    """
    Create a FastAPI application with or without web interface.

    This function creates a FastAPI app with the web interface enabled by default,
    including README integration for better user experience.

    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        env_name: Optional environment name for README loading

    Returns:
        FastAPI application instance with or without web interface and README integration
    """
    # Check if web interface should be enabled
    # This can be controlled via environment variable or build argument
    enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    if enable_web:
        # Import web interface only when needed
        from .web_interface import create_web_interface_app

        return create_web_interface_app(env, action_cls, observation_cls, env_name)
    else:
        # Use standard FastAPI app without web interface
        return create_fastapi_app(env, action_cls, observation_cls)


def create_fastapi_app(
    env: Environment,
    action_cls: Type[Action],
    observation_cls: Type[Observation],
) -> Any:
    """
    Create a FastAPI application with routes for the given environment.

    Args:
        env: The Environment instance to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns

    Returns:
        FastAPI application instance with routes registered

    Example:
        >>> from envs.coding_env.server import CodeExecutionEnvironment
        >>> from envs.coding_env.models import CodeAction, CodeObservation
        >>>
        >>> env = CodeExecutionEnvironment()
        >>> app = create_fastapi_app(env, CodeAction, CodeObservation)
        >>>
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(title="Environment HTTP Server")
    server = HTTPEnvServer(env, action_cls, observation_cls)
    server.register_routes(app)
    return app
