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
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar

from .client_types import StepResult, StateT
from .containers.runtime import LocalDockerProvider, UVProvider
from .utils import convert_to_ws_url

if TYPE_CHECKING:
    from .containers.runtime import ContainerProvider, RuntimeProvider
    from websockets.sync.client import ClientConnection

from websockets.sync.client import connect as ws_connect

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
EnvClientT = TypeVar("EnvClientT", bound="EnvClient")


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    """
    Environment client for persistent sessions.

    This client maintains a persistent WebSocket connection to an environment
    server, enabling efficient multi-step interactions. Each client instance
    corresponds to a dedicated environment session on the server.

    Features:
    - Lower latency for sequential interactions
    - Session state is maintained server-side
    - Better suited for long-running episodes

    Example:
        >>> from envs.coding_env.client import CodingEnv
        >>>
        >>> # Connect to a server
        >>> with CodingEnv(base_url="ws://localhost:8000") as env:
        ...     result = env.reset(seed=42)
        ...     while not result.done:
        ...         action = agent.predict(result.observation)
        ...         result = env.step(action)
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
        provider: Optional["ContainerProvider | RuntimeProvider"] = None,
    ):
        """
        Initialize environment client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
                     Will be converted to ws:// if http:// is provided.
            connect_timeout_s: Timeout for establishing WebSocket connection
            message_timeout_s: Timeout for receiving responses to messages
            provider: Optional container/runtime provider for lifecycle management.
                     Can be a ContainerProvider (Docker) or RuntimeProvider (UV).
        """
        # Convert HTTP URL to WebSocket URL
        ws_url = convert_to_ws_url(base_url)

        self._ws_url = f"{ws_url}/ws"
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._provider = provider
        self._ws: Optional[ClientConnection] = None

    def connect(self) -> "EnvClient":
        """
        Establish WebSocket connection to the server.

        Returns:
            self for method chaining

        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._ws is not None:
            return self

        try:
            self._ws = ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e

        return self

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                # Send close message
                self._send({"type": "close"})
            except Exception:
                pass  # Best effort
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _ensure_connected(self) -> None:
        """Ensure WebSocket connection is established."""
        if self._ws is None:
            self.connect()

    def _send(self, message: Dict[str, Any]) -> None:
        """Send a message over the WebSocket."""
        self._ensure_connected()
        assert self._ws is not None
        self._ws.send(json.dumps(message))

    def _receive(self) -> Dict[str, Any]:
        """Receive and parse a message from the WebSocket."""
        assert self._ws is not None
        raw = self._ws.recv(timeout=self._message_timeout)
        return json.loads(raw)

    def _send_and_receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and wait for response."""
        self._send(message)
        response = self._receive()

        # Check for error response
        if response.get("type") == "error":
            error_data = response.get("data", {})
            raise RuntimeError(
                f"Server error: {error_data.get('message', 'Unknown error')} "
                f"(code: {error_data.get('code', 'UNKNOWN')})"
            )

        return response

    @classmethod
    def from_docker_image(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> EnvClientT:
        """
        Create an environment client by spinning up a Docker container.

        Args:
            image: Docker image name to run (e.g., "coding-env:latest")
            provider: Container provider to use (defaults to LocalDockerProvider)
            **kwargs: Additional arguments to pass to provider.start_container()

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
        client.connect()

        return client

    @classmethod
    def from_hub(
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
            repo_id: Hugging Face space identifier ``{org}/{space}``.
            use_docker: When ``True`` (default) pull from the HF registry and
                launch via :class:`LocalDockerProvider`. When ``False`` run the
                space locally with :class:`UVProvider`.
            provider: Optional provider instance to reuse. Must be a
                :class:`ContainerProvider` when ``use_docker=True`` and a
                :class:`RuntimeProvider` otherwise.
            provider_kwargs: Additional keyword arguments forwarded to
                either the container provider's ``start_container`` (docker)
                or to the ``UVProvider`` constructor/start (uv). When
                ``use_docker=False``, the ``project_path`` argument can be
                used to override the default git URL
                (``git+https://huggingface.co/spaces/{repo_id}``).

        Returns:
            Connected client instance

        Examples:
            >>> # Pull and run from HF Docker registry
            >>> env = MyEnv.from_hub("openenv/echo-env")
            >>>
            >>> # Run locally with UV (clones the space)
            >>> env = MyEnv.from_hub("openenv/echo-env", use_docker=False)
            >>>
            >>> # Run from a local checkout
            >>> env = MyEnv.from_hub(
            ...     "openenv/echo-env",
            ...     use_docker=False,
            ...     project_path="/path/to/local/checkout"
            ... )
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
            client.connect()
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

            base_url = provider.start(**start_args)
            provider.wait_for_ready()

            client = cls(base_url=base_url, provider=provider)
            client.connect()
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

    def reset(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment with optional parameters.

        Args:
            **kwargs: Optional parameters passed to the environment's reset method.
                     Common parameters include:
                     - seed: Random seed for reproducibility
                     - episode_id: Custom episode identifier

        Returns:
            StepResult containing initial observation
        """
        message = {
            "type": "reset",
            "data": kwargs,
        }
        response = self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute
            **kwargs: Optional parameters (currently ignored)

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    def state(self) -> StateT:
        """
        Get the current environment state from the server.

        Returns:
            State object with environment state information
        """
        message = {"type": "state"}
        response = self._send_and_receive(message)
        return self._parse_state(response.get("data", {}))

    def close(self) -> None:
        """
        Close the WebSocket connection and clean up resources.

        If this client was created via from_docker_image() or from_hub(),
        this will also stop and remove the associated container/process.
        """
        self.disconnect()

        if self._provider is not None:
            # Handle both ContainerProvider and RuntimeProvider
            if hasattr(self._provider, "stop_container"):
                self._provider.stop_container()
            elif hasattr(self._provider, "stop"):
                self._provider.stop()

    def __enter__(self) -> "EnvClient":
        """Enter context manager, ensuring connection is established."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing connection."""
        self.close()
