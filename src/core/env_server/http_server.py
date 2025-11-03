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

import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Type

from .interfaces import Environment
from .types import Action, Observation
from fastapi import Body, FastAPI

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
            observation = self.env.reset()
            return self._serialize_observation(observation)

        @app.post("/step")
        async def step(request: Dict[str, Any]) -> Dict[str, Any]:
            """Step endpoint - executes action and returns observation."""
            action_data = request.get("action", {})
            
            # Extract timeout_s from request (sent by HTTPEnvClient)
            timeout_s = request.get("timeout_s", None)
            
            # TODO: Handle request_id, episode_id from request if provided

            # Deserialize action
            action = self._deserialize_action(action_data)

            # Execute step with timeout if environment supports it
            try:
                # Try to pass timeout_s to step() method
                observation = self.env.step(action, timeout_s=timeout_s)
            except TypeError:
                # Environment doesn't support timeout parameter, call without it
                observation = self.env.step(action)

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


    def _deserialize_action(self, action_data: Dict[str, Any]) -> Action:
        """
        Convert JSON dict to Action instance.

        Args:
            action_data: Dictionary containing action data

        Returns:
            Action instance

        Note:
            This is a simple implementation. Subclasses may need to override
            for more complex deserialization logic.
        """
        # Remove metadata if present (it will be set via kw_only field)
        metadata = action_data.pop("metadata", {})
        action = self.action_cls(**action_data)
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

        # Extract reward and done (these are part of StepResult on client side)
        reward = obs_dict.pop("reward", None)
        done = obs_dict.pop("done", False)
        obs_dict.pop("metadata", None)  # Remove metadata from observation

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
    enable_web = (
        os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
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
