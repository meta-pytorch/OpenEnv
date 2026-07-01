# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment client for persistent sessions.

This module provides a WebSocket-based client that maintains a persistent connection
to an environment server, enabling efficient multi-step interactions without
the overhead of HTTP request/response cycles.

The client is async by default. For synchronous usage, use the `.sync()` method
to get a `SyncEnvClient` wrapper.

Examples:

    Async usage:

    ```python
    async with GenericEnvClient(base_url="ws://localhost:8000") as env:
        result = await env.reset()
        result = await env.step({"code": "print('hello')"})
    ```

    Sync usage via `.sync()` wrapper:

    ```python
    env = GenericEnvClient(base_url="ws://localhost:8000").sync()
    with env:
        result = env.reset()
        result = env.step({"code": "print('hello')"})
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar
from urllib.parse import urlsplit

from .client_types import StateT, StepResult
from .containers.runtime import LocalDockerProvider, UVProvider
from .utils import convert_to_ws_url

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    from .containers.runtime import ContainerProvider, RuntimeProvider
    from .sync_client import SyncEnvClient

from websockets.asyncio.client import connect as ws_connect

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
EnvClientT = TypeVar("EnvClientT", bound="EnvClient")

_VALID_CLIENT_MODES = ("simulation", "production")


def _normalize_mode(mode: Optional[str]) -> str:
    """Resolve and validate the client communication mode."""
    raw_mode = (
        os.environ.get("OPENENV_CLIENT_MODE", "simulation") if mode is None else mode
    )
    normalized_mode = raw_mode.lower()
    if normalized_mode not in _VALID_CLIENT_MODES:
        raise ValueError(
            f"Invalid mode: '{normalized_mode}'. Must be 'simulation' or 'production'. "
            f"Set via constructor parameter or OPENENV_CLIENT_MODE environment variable."
        )
    return normalized_mode


def _is_localhost_ws_url(ws_url: str) -> bool:
    """Return True when the WebSocket URL targets the local loopback interface.

    The hostname is parsed from the URL so that only the actual host is matched.
    Substring matching is avoided because remote hosts such as
    ``my-localhost-proxy.example.com`` or ``127.0.0.1.example.com`` must not be
    treated as local.
    """
    hostname = urlsplit(ws_url).hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _required_start_container_parameters(provider: Any) -> list[str]:
    """Return required arguments for a bound provider.start_container()."""
    try:
        signature = inspect.signature(provider.start_container)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """
    Async environment client for persistent sessions.

    This client maintains a persistent WebSocket connection to an environment
    server, enabling efficient multi-step interactions. Each client instance
    corresponds to a dedicated environment session on the server.

    The client is async by default. For synchronous usage, use the `.sync()`
    method to get a `SyncEnvClient` wrapper.

    Features:
    - Lower latency for sequential interactions
    - Session state is maintained server-side
    - Better suited for long-running episodes
    - Async by default for modern Python async/await patterns

    Examples:

        Async usage:

        ```python
        from envs.coding_env.client import CodingEnv

        async with CodingEnv(base_url="ws://localhost:8000") as env:
            result = await env.reset(seed=42)
            while not result.done:
                action = agent.predict(result.observation)
                result = await env.step(action)
        ```

        Sync usage via `.sync()` wrapper:

        ```python
        env = CodingEnv(base_url="ws://localhost:8000").sync()
        with env:
            result = env.reset(seed=42)
            result = env.step(action)
        ```
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        max_message_size_mb: float = 100.0,
        websocket_ping_interval_s: Optional[float] = 20.0,
        websocket_ping_timeout_s: Optional[float] = 20.0,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        mode: Optional[str] = None,
    ):
        """
        Initialize environment client.

        Args:
            base_url (`str`, *optional*):
                Base URL of the environment server (http:// or ws://). Will be converted to
                ws:// if http:// is provided. May be omitted when the provider
                has enough constructor state to start itself.
            connect_timeout_s (`float`, *optional*, defaults to `10.0`):
                Timeout for establishing WebSocket connection.
            message_timeout_s (`float`, *optional*, defaults to `60.0`):
                Timeout for receiving responses to messages.
            max_message_size_mb (`float`, *optional*, defaults to `100.0`):
                Maximum WebSocket message size in megabytes. Default 100MB to handle large
                observations (screenshots, DOM, etc.).
            websocket_ping_interval_s (`float` or `None`, *optional*, defaults to `20.0`):
                WebSocket keepalive ping interval. Pass `None` to disable.
            websocket_ping_timeout_s (`float` or `None`, *optional*, defaults to `20.0`):
                WebSocket keepalive pong timeout. Pass `None` to disable.
            provider (`ContainerProvider` or `RuntimeProvider`, *optional*):
                Container/runtime provider for lifecycle management.
            mode (`str`, *optional*):
                Communication mode: `'simulation'` for Gym-style API (default) or
                `'production'` for MCP JSON-RPC protocol. Can also be set via the
                `OPENENV_CLIENT_MODE` environment variable. Constructor parameter takes
                precedence over environment variable. Case-insensitive.
        """
        if base_url is None and provider is None:
            raise ValueError("EnvClient requires either base_url or provider.")

        # Store mode (use object.__setattr__ to bypass immutability)
        object.__setattr__(self, "_mode", _normalize_mode(mode))

        self._base_url: Optional[str] = None
        self._ws_url: Optional[str] = None
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._max_message_size = int(
            max_message_size_mb * 1024 * 1024
        )  # Convert MB to bytes
        self._websocket_ping_interval_s = websocket_ping_interval_s
        self._websocket_ping_timeout_s = websocket_ping_timeout_s
        self._provider = provider
        self._start_provider_on_connect = base_url is None
        self._child_clients: list[EnvClient[Any, Any, Any]] = []
        self._ws: Optional[ClientConnection] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        if base_url is not None:
            self._set_base_url(base_url)

    def _set_base_url(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        ws_url = convert_to_ws_url(base_url)
        self._ws_url = f"{ws_url}/ws"

    def _start_provider_if_needed(self) -> None:
        if self._ws_url is not None:
            return
        if self._provider is None:
            raise RuntimeError("EnvClient has no base URL or provider.")
        if hasattr(self._provider, "start_container"):
            required_parameters = _required_start_container_parameters(self._provider)
            if required_parameters:
                required = ", ".join(required_parameters)
                raise ValueError(
                    f"{type(self._provider).__name__} does not support "
                    "provider-owned startup because start_container() requires "
                    f"{required}. Start the provider manually and pass base_url, "
                    "or configure a provider with a constructor-owned image/source."
                )
            base_url = self._provider.start_container()
            self._provider.wait_for_ready(base_url)
        elif hasattr(self._provider, "start"):
            base_url = self._provider.start()
            self._provider.wait_for_ready()
        else:
            raise TypeError("provider must define start_container() or start().")
        self._set_base_url(base_url)

    def _create_session_client(self) -> "EnvClient[Any, Any, Any]":
        self._start_provider_if_needed()
        if self._base_url is None:
            raise RuntimeError("EnvClient has no base URL.")

        signature = inspect.signature(type(self))
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        candidate_kwargs = {
            "base_url": self._base_url,
            "connect_timeout_s": self._connect_timeout,
            "message_timeout_s": self._message_timeout,
            "max_message_size_mb": self._max_message_size / (1024 * 1024),
            "websocket_ping_interval_s": self._websocket_ping_interval_s,
            "websocket_ping_timeout_s": self._websocket_ping_timeout_s,
            "mode": self._mode,
        }
        constructor_kwargs = {}
        for name, value in candidate_kwargs.items():
            if accepts_kwargs or name in signature.parameters:
                constructor_kwargs[name] = value

        client = type(self)(**constructor_kwargs)
        return client

    async def new_session(self) -> "EnvClient[Any, Any, Any]":
        """
        Create and connect a new session against the same environment server.

        Returns:
            `EnvClient`: A connected child client of the same concrete type.

        The child session is tracked by this parent and closed when the parent
        is closed. Server-side capacity still applies: when the server is at
        `MAX_CONCURRENT_ENVS`, opening the child WebSocket can fail and is
        surfaced as a connection error.
        """
        client = self._create_session_client()
        await client.connect()
        self._child_clients.append(client)
        return client

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification of _mode after initialization."""
        if name == "_mode" and hasattr(self, "_mode"):
            raise AttributeError("Cannot modify mode after initialization")
        super().__setattr__(name, value)

    async def connect(self) -> "EnvClient":
        """
        Establish WebSocket connection to the server.

        Returns:
            self for method chaining

        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._ws is not None:
            if self._ws_loop is asyncio.get_running_loop():
                return self
            # Connected from a different event loop than the one running
            # now -- e.g. `client = await Client.from_env(...)` inside
            # `asyncio.run(...)`, then `client.sync()` drives every later
            # call on `SyncEnvClient`'s own dedicated background loop. The
            # websocket object is bound to internals of the original loop,
            # which is typically already closed by the time we get here, so
            # it cannot be reused (or even cleanly closed) from this loop.
            # Drop the stale reference and reconnect fresh below rather than
            # silently no-op-ing onto a dead connection.
            self._ws = None
            self._ws_loop = None

        try:
            self._start_provider_if_needed()
        except Exception:
            await self.close()
            raise

        assert self._ws_url is not None

        # Disable the proxy for localhost connections via the per-connection
        # `proxy` argument rather than mutating the process-global NO_PROXY
        # env var: concurrent connect() calls (e.g. asyncio.gather over many
        # env clients) would otherwise race on os.environ and leak state.
        connect_kwargs: Dict[str, Any] = {}
        if _is_localhost_ws_url(self._ws_url):
            connect_kwargs["proxy"] = None

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=self._websocket_ping_interval_s,
                ping_timeout=self._websocket_ping_timeout_s,
                **connect_kwargs,
            )
            self._ws_loop = asyncio.get_running_loop()
        except Exception as e:
            await self.close()
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e

        return self

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            ws = self._ws
            ws_loop = self._ws_loop
            same_loop = ws_loop is asyncio.get_running_loop()
            try:
                if same_loop:
                    await ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass  # Best effort
            try:
                if same_loop:
                    await ws.close()
            except Exception:
                pass
            self._ws = None
            self._ws_loop = None

    async def _ensure_connected(self) -> None:
        """Ensure WebSocket connection is established on the current loop.

        Always delegates to `connect()` rather than pre-checking `self._ws is
        None`: `connect()` itself is the one that knows whether an existing
        `_ws` is reusable (same event loop) or stale (a different one, e.g.
        from a prior `from_env()` call now being driven through `.sync()`'s
        own loop). A pre-check here that only looked at `_ws is None` would
        skip `connect()` entirely whenever `_ws` is already set -- including
        the stale-loop case -- so the reconnect logic would never run for
        callers that never explicitly call `.connect()` themselves (e.g.
        `client.sync().reset()` right after `from_env()`).
        """
        await self.connect()

    async def _send(self, message: Dict[str, Any]) -> None:
        """Send a message over the WebSocket."""
        await self._ensure_connected()
        assert self._ws is not None
        await self._ws.send(json.dumps(message))

    async def _receive(self) -> Dict[str, Any]:
        """Receive and parse a message from the WebSocket."""
        assert self._ws is not None
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._message_timeout)
        return json.loads(raw)

    async def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""
        await self._send(message)
        response = await self._receive()

        # Check for error response
        if response.get("type") == "error":
            error_data = response.get("data", {})
            raise RuntimeError(
                f"Server error: {error_data.get('message', 'Unknown error')} "
                f"(code: {error_data.get('code', 'UNKNOWN')})"
            )

        return response

    @classmethod
    async def from_docker_image(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> EnvClientT:
        """
        Create an environment client by spinning up a Docker container.

        Args:
            image (`str`):
                Docker image name to run (e.g., `"coding-env:latest"`).
            provider (`ContainerProvider`, *optional*):
                Container provider to use. Defaults to `LocalDockerProvider`.
            **kwargs:
                Additional arguments to pass to `provider.start_container()`.

        Returns:
            Connected client instance
        """
        if provider is None:
            provider = LocalDockerProvider()

        # Start container
        base_url = provider.start_container(image, **kwargs)

        # Wait for server to be ready
        provider.wait_for_ready(base_url)

        # Create and connect client
        client = cls(base_url=base_url, provider=provider)
        await client.connect()

        return client

    @classmethod
    async def from_env(
        cls: Type[EnvClientT],
        repo_id: str,
        *,
        use_docker: bool = True,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
        **provider_kwargs: Any,
    ) -> EnvClientT:
        """
        Create a client from a Hugging Face Space.

        Args:
            repo_id (`str`):
                Hugging Face space identifier `{org}/{space}`.
            use_docker (`bool`, *optional*, defaults to `True`):
                When `True`, pull from the HF registry and launch via `LocalDockerProvider`.
                When `False`, run the space locally with `UVProvider`.
            provider (`ContainerProvider` or `RuntimeProvider`, *optional*):
                Provider instance to reuse. Must be a `ContainerProvider` when
                `use_docker=True` and a `RuntimeProvider` otherwise.
            **provider_kwargs:
                Additional keyword arguments forwarded to either the container provider's
                `start_container` (docker) or to the `UVProvider` constructor/start (uv).
                When `use_docker=False`, the `project_path` argument can be used to override
                the default git URL (`git+https://huggingface.co/spaces/{repo_id}`).

        Returns:
            Connected client instance

        Examples:

            ```python
            # Pull and run from HF Docker registry
            env = await MyEnv.from_env("openenv/echo-env")

            # Run locally with UV (clones the space)
            env = await MyEnv.from_env("openenv/echo-env", use_docker=False)

            # Run from a local checkout
            env = await MyEnv.from_env(
                "openenv/echo-env",
                use_docker=False,
                project_path="/path/to/local/checkout"
            )
            ```
        """
        # Extract start args that apply to both providers
        start_args = {}
        for key in ("port", "env_vars", "workers"):
            if key in provider_kwargs:
                start_args[key] = provider_kwargs.pop(key)

        if use_docker:
            # Docker mode: pull from HF registry
            docker_provider = provider or LocalDockerProvider()
            tag = provider_kwargs.pop("tag", "latest")
            image = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"
            base_url = docker_provider.start_container(
                image, **start_args, **provider_kwargs
            )
            docker_provider.wait_for_ready(base_url)

            client = cls(base_url=base_url, provider=docker_provider)
            await client.connect()
            return client
        else:
            # UV mode: clone and run with uv
            if provider is None:
                uv_kwargs = dict(provider_kwargs)
                project_path = uv_kwargs.pop("project_path", None)
                if project_path is None:
                    project_path = f"git+https://huggingface.co/spaces/{repo_id}"

                provider = UVProvider(project_path=project_path, **uv_kwargs)
            else:
                if provider_kwargs:
                    raise ValueError(
                        "provider_kwargs cannot be used when supplying a provider instance"
                    )

            try:
                context_timeout_s = getattr(provider, "context_timeout_s", None)
                deadline = (
                    time.monotonic() + context_timeout_s
                    if context_timeout_s is not None
                    else None
                )
                base_url = provider.start(**start_args)
                if deadline is None:
                    provider.wait_for_ready()
                else:
                    provider.wait_for_ready(
                        timeout_s=max(0.0, deadline - time.monotonic())
                    )

                client = cls(base_url=base_url, provider=provider)
                await client.connect()
            except Exception:
                # No EnvClient may exist yet for the caller to close(), so
                # this is the only chance to release the spawned process and
                # (for a git+ project_path) the temp clone directory.
                provider.stop()
                raise
            return client

    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        """Convert an Action object to the JSON data expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        """Convert a JSON response from the state endpoint to a State object."""
        raise NotImplementedError

    async def reset(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment with optional parameters.

        Args:
            **kwargs:
                Optional parameters passed to the environment's reset method.

        Returns:
            StepResult containing initial observation
        """
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    async def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action:
                The action to execute.
            **kwargs:
                Optional parameters (currently ignored).

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = await self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    async def state(self) -> StateT:
        """
        Get the current environment state from the server.

        Returns:
            State object with environment state information
        """
        message = {"type": "state"}
        response = await self._send_and_receive(message)
        return self._parse_state(response.get("data", {}))

    async def close(self) -> None:
        """
        Close the WebSocket connection and clean up resources.

        If this client was created via from_docker_image() or from_env(),
        this will also stop and remove the associated container/process.
        """
        for child in list(self._child_clients):
            with suppress(Exception):
                await child.close()
        self._child_clients.clear()

        try:
            await self.disconnect()
        finally:
            try:
                if self._provider is not None:
                    # Handle both ContainerProvider and RuntimeProvider
                    if hasattr(self._provider, "stop_container"):
                        self._provider.stop_container()
                    elif hasattr(self._provider, "stop"):
                        self._provider.stop()
            finally:
                if self._start_provider_on_connect:
                    self._base_url = None
                    self._ws_url = None

    async def __aenter__(self) -> "EnvClient":
        """Enter async context manager, ensuring connection is established."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing connection."""
        await self.close()

    def __enter__(self) -> "EnvClient":
        """Sync context manager entry - raises error suggesting async usage."""
        raise TypeError(
            "EnvClient is async by default. Use 'async with' instead of 'with', "
            "or call .sync() to get a synchronous wrapper:\n"
            "  async with client:  # async usage\n"
            "  with client.sync():  # sync wrapper"
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sync context manager exit - should not be reached."""
        pass  # pragma: no cover

    def sync(self) -> "SyncEnvClient":
        """
        Return a synchronous wrapper around this async client.

        Use this method when you need synchronous access to the environment
        without async/await syntax. This is useful for:
        - Integration with synchronous codebases
        - Interactive/REPL usage
        - Stopping async from "infecting" the call stack

        Returns:
            SyncEnvClient wrapper that provides synchronous methods

        Examples:

            ```python
            async_client = GenericEnvClient(base_url="http://localhost:8000")
            sync_client = async_client.sync()

            with sync_client:
                result = sync_client.reset()
                result = sync_client.step({"code": "print('hello')"})
            ```
        """
        from .sync_client import SyncEnvClient

        return SyncEnvClient(self)
