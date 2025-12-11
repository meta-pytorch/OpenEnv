# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HTTP server wrapper for Environment instances.

This module provides utilities to wrap any Environment subclass and expose it
over HTTP endpoints that HTTPEnvClient can consume. Also supports WebSocket
connections for persistent sessions with multi-environment concurrency.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Type, Union

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from .interfaces import Environment
from .route_config import (
    GetEndpointConfig,
    register_get_endpoints,
)
from .serialization import deserialize_action, serialize_observation
from .types import (
    Action,
    Observation,
    ResetRequest,
    ResetResponse,
    State,
    StepRequest,
    StepResponse,
    EnvironmentMetadata,
    SchemaResponse,
    HealthResponse,
    WSResetMessage,
    WSStepMessage,
    WSStateMessage,
    WSCloseMessage,
    WSObservationResponse,
    WSStateResponse,
    WSErrorResponse,
    ConcurrencyConfig,
    ServerCapacityStatus,
    SessionInfo,
)
from .exceptions import (
    ConcurrencyConfigurationError,
    SessionCapacityError,
    EnvironmentFactoryError,
)


class HTTPEnvServer:
    """
    HTTP server wrapper for Environment instances.

    This class wraps an Environment and exposes its reset(), step(), and state
    methods as HTTP endpoints compatible with HTTPEnvClient. Also supports
    WebSocket connections for persistent sessions with multi-environment concurrency.

    The server expects:
    - Action deserialization: Converts JSON dict to Action subclass
    - Observation serialization: Converts Observation subclass to JSON dict

    Example:
        >>> from core.env_server import HTTPEnvServer
        >>> from envs.coding_env.server import CodeExecutionEnvironment
        >>>
        >>> # Single environment (backward compatible)
        >>> env = CodeExecutionEnvironment()
        >>> server = HTTPEnvServer(env)
        >>>
        >>> # Factory pattern for concurrent sessions
        >>> server = HTTPEnvServer(
        ...     env=CodeExecutionEnvironment,  # Pass class, not instance
        ...     max_concurrent_envs=4,
        ... )
        >>>
        >>> # Register routes with FastAPI
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> server.register_routes(app)
    """

    def __init__(
        self,
        env: Union[Environment, Callable[[], Environment], Type[Environment]],
        action_cls: Type[Action] = None,
        observation_cls: Type[Observation] = None,
        max_concurrent_envs: int = 1,
        skip_concurrency_check: bool = False,
        concurrency_config: Optional[ConcurrencyConfig] = None,
    ):
        """
        Initialize HTTP server wrapper.

        Args:
            env: The Environment instance, factory callable, or class to wrap.
                 - If an instance is provided, it's used directly (single-env mode)
                 - If a callable/class is provided, it's called to create new
                   environments for each WebSocket session (factory mode)
            action_cls: The Action subclass this environment expects
            observation_cls: The Observation subclass this environment returns
            max_concurrent_envs: Maximum number of concurrent WebSocket sessions.
                                 Only applies when env is a factory. Default is 1.
                                 If concurrency_config is provided, this parameter is ignored.
            skip_concurrency_check: If True, skip concurrency safety validation.
                                    Use with caution for advanced users who understand
                                    the isolation requirements.
            concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                                If provided, overrides max_concurrent_envs and allows
                                configuration of session timeout and capacity behavior.
                                    
        Raises:
            ConcurrencyConfigurationError: If max_concurrent_envs > 1 for an
                environment that is not marked as CONCURRENCY_SAFE.
        """
        self._env_factory: Optional[Callable[[], Environment]] = None
        
        # Handle concurrency configuration
        if concurrency_config is not None:
            self._concurrency_config = concurrency_config
            self._max_concurrent_envs = concurrency_config.max_concurrent_envs
        else:
            # Use legacy parameters
            self._concurrency_config = ConcurrencyConfig(
                max_concurrent_envs=max_concurrent_envs,
                session_timeout_seconds=None,
                reject_on_capacity=True,
            )
            self._max_concurrent_envs = max_concurrent_envs
        
        self._skip_concurrency_check = skip_concurrency_check or os.getenv(
            "OPENENV_SKIP_CONCURRENCY_CHECK", ""
        ).lower() in ("1", "true", "yes")
        
        # Determine if env is an instance or factory
        if isinstance(env, Environment):
            # Single instance mode (backward compatible)
            self.env = env
            self._env_factory = None
        elif callable(env):
            # Factory mode - env is a class or callable
            self._env_factory = env
            # Create a single instance for HTTP endpoints (backward compat)
            try:
                self.env = env()
            except Exception as e:
                factory_name = getattr(env, "__name__", str(env))
                raise EnvironmentFactoryError(factory_name, e) from e
        else:
            raise TypeError(
                f"env must be an Environment instance or callable, got {type(env)}"
            )
        
        # Validate concurrency configuration
        self._validate_concurrency_safety()
        
        self.action_cls = action_cls
        self.observation_cls = observation_cls
        
        # Session management for WebSocket connections
        self._sessions: Dict[str, Environment] = {}
        self._session_executors: Dict[str, ThreadPoolExecutor] = {}
        self._session_info: Dict[str, SessionInfo] = {}
        self._session_lock = asyncio.Lock()
        
        # Create thread pool for running sync code in async context
        # This is needed for environments using sync libraries (e.g., Playwright)
        # Configurable via OPENENV_THREAD_POOL_SIZE (default: 32)
        pool_size = int(os.getenv("OPENENV_THREAD_POOL_SIZE", "32"))
        self._executor = ThreadPoolExecutor(max_workers=pool_size)

        # Check if environment has async methods for better concurrency
        self._has_step_async = hasattr(env, "step_async") and asyncio.iscoroutinefunction(env.step_async)
        self._has_reset_async = hasattr(env, "reset_async") and asyncio.iscoroutinefunction(env.reset_async)

    def _validate_concurrency_safety(self) -> None:
        """
        Validate that the environment supports the configured concurrency level.
        
        Raises:
            ConcurrencyConfigurationError: If max_concurrent_envs > 1 for an
                environment that is not marked as CONCURRENCY_SAFE.
        """
        if self._max_concurrent_envs <= 1:
            return
        
        if self._skip_concurrency_check:
            return
        
        is_concurrency_safe = getattr(self.env, "CONCURRENCY_SAFE", False)
        
        if not is_concurrency_safe:
            env_name = type(self.env).__name__
            raise ConcurrencyConfigurationError(
                environment_name=env_name,
                max_concurrent_envs=self._max_concurrent_envs,
            )

    def get_capacity_status(self) -> ServerCapacityStatus:
        """
        Get the current capacity status of the server.
        
        Returns:
            ServerCapacityStatus with current session counts and availability.
        """
        return ServerCapacityStatus.from_counts(
            active=len(self._sessions),
            max_sessions=self._max_concurrent_envs,
        )

    async def _run_sync_in_thread_pool(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))

    def _get_valid_kwargs(self, sig, kwargs, skip_params=None):
        """Filter kwargs to only include parameters accepted by the function signature."""
        if skip_params is None:
            skip_params = set()

        valid_kwargs = {}

        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        for k, v in kwargs.items():
            if k in sig.parameters or has_kwargs:
                if k not in skip_params:
                    valid_kwargs[k] = v

        return valid_kwargs

    async def _create_session(self) -> tuple[str, Environment]:
        """
        Create a new WebSocket session with its own environment instance.
        
        Returns:
            Tuple of (session_id, environment)
            
        Raises:
            SessionCapacityError: If max concurrent sessions reached
            EnvironmentFactoryError: If the factory fails to create an environment
        """
        import time
        
        async with self._session_lock:
            if len(self._sessions) >= self._max_concurrent_envs:
                if self._concurrency_config.reject_on_capacity:
                    raise SessionCapacityError(
                        active_sessions=len(self._sessions),
                        max_sessions=self._max_concurrent_envs,
                    )
                else:
                    # TODO: Implement queuing mechanism when reject_on_capacity=False
                    raise SessionCapacityError(
                        active_sessions=len(self._sessions),
                        max_sessions=self._max_concurrent_envs,
                        message="Session queuing not yet implemented",
                    )
            
            session_id = str(uuid.uuid4())
            current_time = time.time()
            
            if self._env_factory is None:
                # Single instance mode - use shared env (limited concurrency)
                if self._sessions:
                    raise SessionCapacityError(
                        active_sessions=len(self._sessions),
                        max_sessions=1,
                        message="Single instance mode: only one WebSocket session allowed",
                    )
                env = self.env
            else:
                # Factory mode - create new environment
                try:
                    env = self._env_factory()
                except Exception as e:
                    factory_name = getattr(self._env_factory, "__name__", str(self._env_factory))
                    raise EnvironmentFactoryError(factory_name, e) from e
            
            self._sessions[session_id] = env
            
            # Create dedicated executor for this session
            self._session_executors[session_id] = ThreadPoolExecutor(max_workers=1)
            
            # Track session metadata
            self._session_info[session_id] = SessionInfo(
                session_id=session_id,
                created_at=current_time,
                last_activity_at=current_time,
                step_count=0,
                environment_type=type(env).__name__,
            )
            
            return session_id, env
    
    async def _destroy_session(self, session_id: str) -> None:
        """
        Destroy a WebSocket session and cleanup resources.
        
        Args:
            session_id: The session ID to destroy
        """
        async with self._session_lock:
            if session_id in self._sessions:
                env = self._sessions.pop(session_id)
                # Call close() if environment has it
                if hasattr(env, 'close') and callable(env.close):
                    try:
                        env.close()
                    except Exception:
                        pass  # Best effort cleanup
                
            if session_id in self._session_executors:
                executor = self._session_executors.pop(session_id)
                executor.shutdown(wait=False)
            
            # Remove session metadata
            self._session_info.pop(session_id, None)
    
    def _update_session_activity(self, session_id: str, increment_step: bool = False) -> None:
        """
        Update session activity timestamp and optionally increment step count.
        
        Args:
            session_id: The session ID to update
            increment_step: If True, increment the step count
        """
        import time
        
        if session_id in self._session_info:
            self._session_info[session_id].last_activity_at = time.time()
            if increment_step:
                self._session_info[session_id].step_count += 1
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get information about a specific session.
        
        Args:
            session_id: The session ID to query
            
        Returns:
            SessionInfo if the session exists, None otherwise
        """
        return self._session_info.get(session_id)

    async def _run_in_session_executor(
        self, session_id: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Run a synchronous function in the session's thread pool executor."""
        executor = self._session_executors.get(session_id, self._executor)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    
    @property
    def active_sessions(self) -> int:
        """Return the number of active WebSocket sessions."""
        return len(self._sessions)
    
    @property
    def max_concurrent_envs(self) -> int:
        """Return the maximum number of concurrent environments."""
        return self._max_concurrent_envs

    @property
    def is_concurrency_safe(self) -> bool:
        """Return whether the environment is marked as concurrency safe."""
        return getattr(self.env, "CONCURRENCY_SAFE", False)

    @property
    def concurrency_config(self) -> ConcurrencyConfig:
        """Return the concurrency configuration."""
        return self._concurrency_config

    def register_routes(self, app: FastAPI) -> None:
        """
        Register HTTP routes on a FastAPI application.

        Args:
            app: FastAPI application instance
        """

        # Helper function to handle reset endpoint
        async def reset_handler(
            request: ResetRequest = Body(default_factory=ResetRequest),
        ) -> ResetResponse:
            """Reset endpoint - returns initial observation."""
            # Handle optional parameters
            # Start with all fields from the request, including extra ones
            kwargs = request.model_dump(exclude_unset=True)

            # Pass arguments only if environment accepts them
            if self._has_reset_async:
                sig = inspect.signature(self.env.reset_async)
            else:
                sig = inspect.signature(self.env.reset)
            valid_kwargs = self._get_valid_kwargs(sig, kwargs)

            # Use async method if available for better concurrency
            if self._has_reset_async:
                observation = await self.env.reset_async(**valid_kwargs)
            else:
                observation = await self._run_sync_in_thread_pool(self.env.reset, **valid_kwargs)
            return ResetResponse(**serialize_observation(observation))

        # Helper function to handle step endpoint
        async def step_handler(request: StepRequest) -> StepResponse:
            """Step endpoint - executes action and returns observation."""
            action_data = request.action

            # Deserialize action with Pydantic validation
            try:
                action = deserialize_action(action_data, self.action_cls)
            except ValidationError as e:
                # Return HTTP 422 with detailed validation errors
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=e.errors())

            # Handle optional parameters
            # Start with all fields from the request, including extra ones, but exclude 'action'
            kwargs = request.model_dump(exclude_unset=True, exclude={"action"})

            # Pass arguments only if environment accepts them
            if self._has_step_async:
                sig = inspect.signature(self.env.step_async)
            else:
                sig = inspect.signature(self.env.step)
            valid_kwargs = self._get_valid_kwargs(sig, kwargs, skip_params={"action"})

            # Use async method if available for better concurrency
            if self._has_step_async:
                observation = await self.env.step_async(action, **valid_kwargs)
            else:
                observation = await self._run_sync_in_thread_pool(self.env.step, action, **valid_kwargs)

            # Return serialized observation
            return StepResponse(**serialize_observation(observation))

        # Register routes using the helpers
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
            return await reset_handler(request)

        @app.post(
            "/step",
            response_model=StepResponse,
            tags=["Environment Control"],
            summary="Execute an action in the environment",
            description="""
Execute an action in the environment and receive the resulting observation.

The action must conform to the environment's action schema, which can be
retrieved from the `/schema` endpoint. If the action is invalid,
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
            return await step_handler(request)

        # Configure and register GET endpoints declaratively
        get_endpoints = [
            GetEndpointConfig(
                path="/state",
                handler=lambda: self.env.state,
                response_model=State,
                tag="State Management",
                summary="Get current environment state",
                description="""
Retrieve the current internal state of the environment.

This endpoint allows inspection of the environment state without modifying it.
The structure of the state object is defined by the environment's State model.
                """,
            ),
            GetEndpointConfig(
                path="/metadata",
                handler=self.env.get_metadata,
                response_model=EnvironmentMetadata,
                tag="Environment Info",
                summary="Get environment metadata",
                description="""
Get metadata about this environment.

Returns information about the environment including name, description,
version, author, and documentation links.
                """,
            ),
            GetEndpointConfig(
                path="/health",
                handler=lambda: HealthResponse(status="healthy"),
                response_model=HealthResponse,
                tag="Health",
                summary="Health check",
                description="Check if the environment server is running and healthy.",
            ),
        ]
        register_get_endpoints(app, get_endpoints)

        # Register concurrency config endpoint
        @app.get(
            "/concurrency",
            response_model=ConcurrencyConfig,
            tags=["Environment Info"],
            summary="Get concurrency configuration",
            description="""
Get the current concurrency configuration for this server.

Returns information about:
- **max_concurrent_envs**: Maximum number of concurrent WebSocket sessions
- **session_timeout_seconds**: Timeout for inactive sessions (None if no timeout)
- **reject_on_capacity**: Whether to reject or queue connections at capacity
            """,
        )
        async def get_concurrency_config() -> ConcurrencyConfig:
            """Return concurrency configuration."""
            return self._concurrency_config

        # Register combined schema endpoint
        @app.get(
            "/schema",
            response_model=SchemaResponse,
            tags=["Schema"],
            summary="Get all JSON schemas",
            description="""
Get JSON schemas for actions, observations, and state in a single response.

Returns a combined schema object containing:
- **action**: JSON schema for actions accepted by this environment
- **observation**: JSON schema for observations returned by this environment  
- **state**: JSON schema for environment state objects

This is more efficient than calling individual schema endpoints and provides
all schema information needed to interact with the environment.
            """,
            responses={
                200: {
                    "description": "Combined schemas retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": {
                                "action": {
                                    "type": "object",
                                    "properties": {"message": {"type": "string"}},
                                },
                                "observation": {
                                    "type": "object",
                                    "properties": {"response": {"type": "string"}},
                                },
                                "state": {
                                    "type": "object",
                                    "properties": {"step_count": {"type": "integer"}},
                                },
                            }
                        }
                    },
                }
            },
        )
        async def get_schemas() -> SchemaResponse:
            """Return all schemas in one response."""
            return SchemaResponse(
                action=self.action_cls.model_json_schema(),
                observation=self.observation_cls.model_json_schema(),
                state=State.model_json_schema(),
            )

        # Register WebSocket endpoint for persistent sessions
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """
            WebSocket endpoint for persistent environment sessions.
            
            Each WebSocket connection gets its own environment instance (when using
            factory mode) or shares the single instance (backward compatible mode).
            
            Message Protocol:
            - Client sends: WSResetMessage | WSStepMessage | WSStateMessage | WSCloseMessage
            - Server responds: WSObservationResponse | WSStateResponse | WSErrorResponse
            """
            await websocket.accept()
            
            session_id = None
            session_env = None
            
            try:
                # Create session with dedicated environment
                session_id, session_env = await self._create_session()
                
                while True:
                    # Receive message from client
                    raw_message = await websocket.receive_text()
                    
                    try:
                        message_dict = json.loads(raw_message)
                    except json.JSONDecodeError as e:
                        error_resp = WSErrorResponse(
                            data={"message": f"Invalid JSON: {e}", "code": "INVALID_JSON"}
                        )
                        await websocket.send_text(error_resp.model_dump_json())
                        continue
                    
                    msg_type = message_dict.get("type", "")
                    
                    try:
                        if msg_type == "reset":
                            # Parse and validate reset message
                            try:
                                msg = WSResetMessage(**message_dict)
                            except ValidationError as e:
                                error_resp = WSErrorResponse(
                                    data={"message": "Invalid reset message", "code": "VALIDATION_ERROR", "errors": e.errors()}
                                )
                                await websocket.send_text(error_resp.model_dump_json())
                                continue
                            
                            # Handle reset
                            sig = inspect.signature(session_env.reset)
                            valid_kwargs = self._get_valid_kwargs(sig, msg.data)
                            
                            observation = await self._run_in_session_executor(
                                session_id, session_env.reset, **valid_kwargs
                            )
                            
                            self._update_session_activity(session_id)
                            
                            response = WSObservationResponse(
                                data=serialize_observation(observation)
                            )
                            await websocket.send_text(response.model_dump_json())
                            
                        elif msg_type == "step":
                            # Parse and validate step message
                            try:
                                msg = WSStepMessage(**message_dict)
                            except ValidationError as e:
                                error_resp = WSErrorResponse(
                                    data={"message": "Invalid step message", "code": "VALIDATION_ERROR", "errors": e.errors()}
                                )
                                await websocket.send_text(error_resp.model_dump_json())
                                continue
                            
                            # Deserialize action with Pydantic validation
                            try:
                                action = deserialize_action(msg.data, self.action_cls)
                            except ValidationError as e:
                                error_resp = WSErrorResponse(
                                    data={"message": str(e), "code": "VALIDATION_ERROR", "errors": e.errors()}
                                )
                                await websocket.send_text(error_resp.model_dump_json())
                                continue
                            
                            observation = await self._run_in_session_executor(
                                session_id, session_env.step, action
                            )
                            
                            self._update_session_activity(session_id, increment_step=True)
                            
                            response = WSObservationResponse(
                                data=serialize_observation(observation)
                            )
                            await websocket.send_text(response.model_dump_json())
                            
                        elif msg_type == "state":
                            # Parse and validate state message
                            try:
                                msg = WSStateMessage(**message_dict)
                            except ValidationError as e:
                                error_resp = WSErrorResponse(
                                    data={"message": "Invalid state message", "code": "VALIDATION_ERROR", "errors": e.errors()}
                                )
                                await websocket.send_text(error_resp.model_dump_json())
                                continue
                            
                            # Handle state request
                            state = session_env.state
                            if hasattr(state, 'model_dump'):
                                state_data = state.model_dump()
                            else:
                                state_data = dict(state) if state else {}
                            
                            response = WSStateResponse(data=state_data)
                            await websocket.send_text(response.model_dump_json())
                            
                        elif msg_type == "close":
                            # Parse and validate close message
                            try:
                                msg = WSCloseMessage(**message_dict)
                            except ValidationError as e:
                                error_resp = WSErrorResponse(
                                    data={"message": "Invalid close message", "code": "VALIDATION_ERROR", "errors": e.errors()}
                                )
                                await websocket.send_text(error_resp.model_dump_json())
                                continue
                            
                            # Client requested close
                            break
                            
                        else:
                            error_resp = WSErrorResponse(
                                data={"message": f"Unknown message type: {msg_type}", "code": "UNKNOWN_TYPE"}
                            )
                            await websocket.send_text(error_resp.model_dump_json())
                            
                    except Exception as e:
                        error_resp = WSErrorResponse(
                            data={"message": str(e), "code": "EXECUTION_ERROR"}
                        )
                        await websocket.send_text(error_resp.model_dump_json())
                        
            except WebSocketDisconnect:
                pass
            except SessionCapacityError as e:
                try:
                    error_resp = WSErrorResponse(
                        data={
                            "message": str(e),
                            "code": "CAPACITY_REACHED",
                            "active_sessions": e.active_sessions,
                            "max_sessions": e.max_sessions,
                        }
                    )
                    await websocket.send_text(error_resp.model_dump_json())
                except Exception:
                    pass
            except EnvironmentFactoryError as e:
                try:
                    error_resp = WSErrorResponse(
                        data={
                            "message": str(e),
                            "code": "FACTORY_ERROR",
                            "factory_name": e.factory_name,
                        }
                    )
                    await websocket.send_text(error_resp.model_dump_json())
                except Exception:
                    pass
            except Exception as e:
                try:
                    error_resp = WSErrorResponse(
                        data={"message": str(e), "code": "SESSION_ERROR"}
                    )
                    await websocket.send_text(error_resp.model_dump_json())
                except Exception:
                    pass
            finally:
                # Cleanup session
                if session_id:
                    await self._destroy_session(session_id)
                try:
                    await websocket.close()
                except Exception:
                    pass


def create_app(
    env: Union[Environment, Callable[[], Environment], Type[Environment]],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    env_name: Optional[str] = None,
    max_concurrent_envs: int = 1,
    concurrency_config: Optional[ConcurrencyConfig] = None,
) -> FastAPI:
    """
    Create a FastAPI application with or without web interface.

    This function creates a FastAPI app with the web interface enabled by default,
    including README integration for better user experience.

    Args:
        env: The Environment instance, factory callable, or class to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        env_name: Optional environment name for README loading
        max_concurrent_envs: Maximum concurrent WebSocket sessions (default: 1).
                             Ignored if concurrency_config is provided.
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                            If provided, overrides max_concurrent_envs.

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

        return create_web_interface_app(
            env, action_cls, observation_cls, env_name, 
            max_concurrent_envs, concurrency_config
        )
    else:
        # Use standard FastAPI app without web interface
        return create_fastapi_app(
            env, action_cls, observation_cls, 
            max_concurrent_envs, concurrency_config
        )


def create_fastapi_app(
    env: Union[Environment, Callable[[], Environment], Type[Environment]],
    action_cls: Type[Action],
    observation_cls: Type[Observation],
    max_concurrent_envs: int = 1,
    concurrency_config: Optional[ConcurrencyConfig] = None,
) -> FastAPI:
    """
    Create a FastAPI application with comprehensive documentation.
    
    Args:
        env: The Environment instance, factory callable, or class to serve
        action_cls: The Action subclass this environment expects
        observation_cls: The Observation subclass this environment returns
        max_concurrent_envs: Maximum concurrent WebSocket sessions (default: 1).
                             Ignored if concurrency_config is provided.
        concurrency_config: Optional ConcurrencyConfig for advanced concurrency settings.
                            If provided, overrides max_concurrent_envs.
        
    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

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

    server = HTTPEnvServer(
        env, action_cls, observation_cls, 
        max_concurrent_envs, concurrency_config=concurrency_config
    )
    server.register_routes(app)
    return app
