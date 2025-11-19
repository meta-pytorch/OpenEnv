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
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Type

from fastapi import Body, FastAPI, HTTPException, status
from pydantic import ValidationError

from .interfaces import Environment
from .types import (
    Action,
    Observation,
    ResetRequest,
    ResetResponse,
    State,
    StepRequest,
    StepResponse,
    EnvironmentMetadata,
)


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

        @app.post(
            "/reset",
            response_model=ResetResponse,
            tags=["Environment Control"],
            summary="Reset the environment",
            description="""
Reset the environment to its initial state and return the first observation.

You can optionally provide a seed for reproducibility and an episode_id for tracking.
            """,
            responses={
                200: {
                    "description": "Environment reset successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "observation": {"status": "ready", "data": {}},
                                "reward": None,
                                "done": False,
                            }
                        }
                    },
                }
            },
        )
        async def reset(
            request: ResetRequest = Body(default_factory=ResetRequest),
        ) -> ResetResponse:
            """Reset endpoint - returns initial observation."""
            # Handle optional parameters
            # Start with all fields from the request, including extra ones
            kwargs = request.model_dump(exclude_unset=True)

            # Pass arguments only if environment accepts them
            sig = inspect.signature(self.env.reset)
            valid_kwargs = {}

            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            for k, v in kwargs.items():
                if k in sig.parameters or has_kwargs:
                    valid_kwargs[k] = v

            # Run synchronous reset in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(
                self._executor, lambda: self.env.reset(**valid_kwargs)
            )
            return ResetResponse(**self._serialize_observation(observation))

        @app.post(
            "/step",
            response_model=StepResponse,
            tags=["Environment Control"],
            summary="Execute an action in the environment",
            description="""
Execute an action in the environment and receive the resulting observation.

The action must conform to the environment's action schema, which can be
retrieved from the `/schema/action` endpoint. If the action is invalid,
the endpoint will return HTTP 422 with detailed validation errors.

The response includes:
- **observation**: The environment's response to the action
- **reward**: Optional reward signal (float or None)
- **done**: Boolean indicating if the episode has terminated
            """,
            responses={
                200: {
                    "description": "Action executed successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "observation": {"status": "success", "data": {}},
                                "reward": 1.0,
                                "done": False,
                            }
                        }
                    },
                },
                422: {
                    "description": "Validation error - invalid action format or values",
                    "content": {
                        "application/json": {
                            "example": {
                                "detail": [
                                    {
                                        "type": "string_too_short",
                                        "loc": ["body", "action", "message"],
                                        "msg": "String should have at least 1 character",
                                        "input": "",
                                    }
                                ]
                            }
                        }
                    },
                },
                500: {"description": "Internal server error during action execution"},
            },
        )
        async def step(request: StepRequest) -> StepResponse:
            """Step endpoint - executes action and returns observation."""
            action_data = request.action

            # Deserialize action with Pydantic validation
            try:
                action = self._deserialize_action(action_data)
            except ValidationError as e:
                # Return HTTP 422 with detailed validation errors
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()
                )

            # Handle optional parameters
            # Start with all fields from the request, including extra ones, but exclude 'action'
            kwargs = request.model_dump(exclude_unset=True, exclude={"action"})

            # Pass arguments only if environment accepts them
            sig = inspect.signature(self.env.step)
            valid_kwargs = {}

            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            for k, v in kwargs.items():
                if k in sig.parameters or has_kwargs:
                    valid_kwargs[k] = v

            # Run synchronous step in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            observation = await loop.run_in_executor(
                self._executor, lambda: self.env.step(action, **valid_kwargs)
            )

            # Return serialized observation
            return StepResponse(**self._serialize_observation(observation))

        @app.get(
            "/state",
            response_model=State,
            tags=["State Management"],
            summary="Get current environment state",
            description="""
Retrieve the current internal state of the environment.

This endpoint allows inspection of the environment state without modifying it.
The structure of the state object is defined by the environment's State model.
            """,
        )
        async def get_state() -> State:
            """State endpoint - returns current environment state."""
            return self.env.state

        @app.get(
            "/metadata",
            response_model=EnvironmentMetadata,
            tags=["Environment Info"],
            summary="Get environment metadata",
            description="""
Get metadata about this environment.

Returns information about the environment including name, description,
version, author, and documentation links.
            """,
        )
        async def get_metadata() -> EnvironmentMetadata:
            """
            Get metadata about this environment.

            Returns information about the environment including name, description,
            version, author, and documentation links.
            """
            return self.env.get_metadata()

        @app.get(
            "/health",
            tags=["Health"],
            summary="Health check",
            description="Check if the environment server is running and healthy.",
        )
        async def health() -> Dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.get(
            "/schema/action",
            tags=["Schema"],
            summary="Get action JSON schema",
            description="""
Get JSON schema for actions accepted by this environment.

Returns the complete JSON schema definition for the Action model,
including all field types, constraints, and validation rules.
This schema can be used to validate actions before sending them
to the environment, or to generate forms in web interfaces.
            """,
        )
        async def get_action_schema() -> Dict[str, Any]:
            """
            Get JSON schema for actions accepted by this environment.

            Returns the complete JSON schema definition for the Action model,
            including all field types, constraints, and validation rules.
            This schema can be used to validate actions before sending them
            to the environment, or to generate forms in web interfaces.

            Returns:
                Dict containing JSON Schema
            """
            return self.action_cls.model_json_schema()

        @app.get(
            "/schema/observation",
            tags=["Schema"],
            summary="Get observation JSON schema",
            description="""
Get JSON schema for observations returned by this environment.

Returns the complete JSON schema definition for the Observation model,
including all field types and nested structures. This schema describes
what observations the environment will return after actions are executed.
            """,
        )
        async def get_observation_schema() -> Dict[str, Any]:
            """
            Get JSON schema for observations returned by this environment.

            Returns the complete JSON schema definition for the Observation model,
            including all field types and nested structures. This schema describes
            what observations the environment will return after actions are executed.

            Returns:
                Dict containing JSON Schema
            """
            return self.observation_cls.model_json_schema()

        @app.get(
            "/schema/state",
            tags=["Schema"],
            summary="Get state JSON schema",
            description="""
Get JSON schema for environment state objects.

Returns the complete JSON schema definition for the State model.
This schema describes the internal state representation of the
environment, which can be queried via the /state endpoint.
            """,
        )
        async def get_state_schema() -> Dict[str, Any]:
            """
            Get JSON schema for environment state objects.

            Returns the complete JSON schema definition for the State model.
            This schema describes the internal state representation of the
            environment, which can be queried via the /state endpoint.

            Returns:
                Dict containing JSON Schema
            """
            return State.model_json_schema()

    def _deserialize_action(self, action_data: Dict[str, Any]) -> Action:
        """
        Convert JSON dict to Action instance using Pydantic validation.

        Args:
            action_data: Dictionary containing action data

        Returns:
            Action instance

        Raises:
            ValidationError: If action_data is invalid for the action class

        Note:
            This uses Pydantic's model_validate() for automatic validation.
        """
        # Pydantic handles validation automatically
        action = self.action_cls.model_validate(action_data)
        return action

    def _serialize_observation(self, observation: Observation) -> Dict[str, Any]:
        """
        Convert Observation instance to JSON-compatible dict using Pydantic.

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
        # Use Pydantic's model_dump() for serialization
        obs_dict = observation.model_dump(
            exclude={
                "reward",
                "done",
                "metadata",
            }  # Exclude these from observation dict
        )

        # Extract reward and done directly from the observation
        reward = observation.reward
        done = observation.done

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
    """Create a FastAPI application with comprehensive documentation."""
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="OpenEnv Environment HTTP API",
        version="1.0.0",
        description="""
# OpenEnv Environment HTTP API

HTTP API for interacting with OpenEnv environments through a standardized interface.

## Features

* **Environment Reset**: Initialize or restart episodes
* **Action Execution**: Send actions and receive observations
* **State Inspection**: Query current environment state
* **Schema Access**: Retrieve JSON schemas for actions and observations

## Workflow

1. Call `/reset` to start a new episode and get initial observation
2. Call `/step` repeatedly with actions to interact with environment
3. Episode ends when observation returns `done: true`
4. Call `/state` anytime to inspect current environment state

## Documentation

* **Swagger UI**: Available at `/docs`
* **ReDoc**: Available at `/redoc`
* **OpenAPI Schema**: Available at `/openapi.json`
        """,
        openapi_tags=[
            {
                "name": "Environment Control",
                "description": "Core operations for environment interaction (reset, step)",
            },
            {
                "name": "State Management",
                "description": "Operations for inspecting environment state",
            },
            {
                "name": "Environment Info",
                "description": "Information about the environment",
            },
            {
                "name": "Schema",
                "description": "JSON Schema endpoints for actions, observations, and state",
            },
            {"name": "Health", "description": "Service health and status checks"},
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "OpenEnv Team",
            "url": "https://github.com/meta-pytorch/OpenEnv",
        },
        license_info={
            "name": "BSD-3-Clause",
            "url": "https://github.com/meta-pytorch/OpenEnv/blob/main/LICENSE",
        },
    )

    server = HTTPEnvServer(env, action_cls, observation_cls)
    server.register_routes(app)
    return app
