# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
WebSocket-based environment client for persistent sessions.

This module provides a WebSocket client that maintains a persistent connection
to an environment server, enabling efficient multi-step interactions without
the overhead of HTTP request/response cycles.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar

from .client_types import StepResult
from .containers.runtime import LocalDockerProvider

if TYPE_CHECKING:
    from .containers.runtime import ContainerProvider
    from websockets.sync.client import ClientConnection

try:
    import websockets
    from websockets.sync.client import connect as ws_connect
except ImportError:
    websockets = None  # type: ignore
    ws_connect = None  # type: ignore

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
WSEnvClientT = TypeVar("WSEnvClientT", bound="WebSocketEnvClient")


class WebSocketEnvClient(ABC, Generic[ActT, ObsT]):
    """
    WebSocket-based environment client for persistent sessions.

    This client maintains a persistent WebSocket connection to an environment
    server, enabling efficient multi-step interactions. Each client instance
    corresponds to a dedicated environment session on the server.

    Compared to HTTPEnvClient:
    - Lower latency for sequential interactions
    - Session state is maintained server-side
    - Better suited for long-running episodes

    Example:
        >>> from envs.coding_env.client import CodingEnvWS
        >>>
        >>> # Connect to a server via WebSocket
        >>> with CodingEnvWS(base_url="ws://localhost:8000") as env:
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
        provider: Optional["ContainerProvider"] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            base_url: Base URL of the environment server (http:// or ws://).
                     Will be converted to ws:// if http:// is provided.
            connect_timeout_s: Timeout for establishing WebSocket connection
            message_timeout_s: Timeout for receiving responses to messages
            provider: Optional container provider for lifecycle management
        """
        if websockets is None:
            raise ImportError(
                "websockets library is required for WebSocketEnvClient. "
                "Install with: pip install websockets"
            )

        # Convert HTTP URL to WebSocket URL
        ws_url = base_url.rstrip("/")
        if ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[7:]
        elif ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[8:]
        elif not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
            ws_url = "ws://" + ws_url

        self._ws_url = f"{ws_url}/ws"
        self._connect_timeout = connect_timeout_s
        self._message_timeout = message_timeout_s
        self._provider = provider
        self._ws: Optional[ClientConnection] = None

    def connect(self) -> "WebSocketEnvClient":
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
        cls: Type[WSEnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> WSEnvClientT:
        """
        Create a WebSocket environment client by spinning up a Docker container.

        Args:
            image: Docker image name to run (e.g., "coding-env:latest")
            provider: Container provider to use (defaults to LocalDockerProvider)
            **kwargs: Additional arguments to pass to provider.start_container()

        Returns:
            Connected WebSocket client instance
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
        cls: Type[WSEnvClientT],
        repo_id: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> WSEnvClientT:
        """
        Create a WebSocket client by pulling from a Hugging Face model hub.
        """
        if provider is None:
            provider = LocalDockerProvider()

        tag = kwargs.pop("tag", "latest")
        base_url = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"

        return cls.from_docker_image(image=base_url, provider=provider, **kwargs)

    @abstractmethod
    def _step_payload(self, action: ActT) -> dict:
        """Convert an Action object to the JSON data expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload: dict) -> Any:
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
            **kwargs: Optional parameters (currently ignored for WebSocket)

        Returns:
            StepResult containing observation, reward, and done status
        """
        message = {
            "type": "step",
            "data": self._step_payload(action),
        }
        response = self._send_and_receive(message)
        return self._parse_result(response.get("data", {}))

    def state(self) -> Any:
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

        If this client was created via from_docker_image(), this will also
        stop and remove the associated container.
        """
        self.disconnect()

        if self._provider is not None:
            self._provider.stop_container()

    def __enter__(self) -> "WebSocketEnvClient":
        """Enter context manager, ensuring connection is established."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing connection."""
        self.close()
