# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synchronous wrapper for async EnvClient.

This module provides a SyncEnvClient that wraps an async EnvClient,
allowing synchronous usage while the underlying client uses async I/O.

Example:
    >>> from openenv.core import GenericEnvClient
    >>>
    >>> # Create async client and get sync wrapper
    >>> async_client = GenericEnvClient(base_url="http://localhost:8000")
    >>> sync_client = async_client.sync()
    >>>
    >>> # Use synchronous API
    >>> with sync_client:
    ...     result = sync_client.reset()
    ...     result = sync_client.step({"code": "print('hello')"})
"""

from __future__ import annotations

from typing import Any, Dict, Generic, TYPE_CHECKING, TypeVar

from .client_types import StepResult, StateT
from .utils import run_async_safely

if TYPE_CHECKING:
    from .env_client import EnvClient

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")


class SyncEnvClient(Generic[ActT, ObsT, StateT]):
    """
    Synchronous wrapper around an async EnvClient.

    This class provides a synchronous interface to an async EnvClient,
    making it easier to use in synchronous code or to stop async from
    "infecting" the entire call stack.

    The wrapper uses `run_async_safely()` to execute async operations,
    which handles both sync and async calling contexts correctly.

    Example:
        >>> # From an async client
        >>> async_client = GenericEnvClient(base_url="http://localhost:8000")
        >>> sync_client = async_client.sync()
        >>>
        >>> # Use synchronous context manager
        >>> with sync_client:
        ...     result = sync_client.reset()
        ...     result = sync_client.step({"action": "test"})

    Attributes:
        _async: The wrapped async EnvClient instance
    """

    def __init__(self, async_client: "EnvClient[ActT, ObsT, StateT]"):
        """
        Initialize sync wrapper around an async client.

        Args:
            async_client: The async EnvClient to wrap
        """
        self._async = async_client

    @property
    def async_client(self) -> "EnvClient[ActT, ObsT, StateT]":
        """Access the underlying async client."""
        return self._async

    def connect(self) -> "SyncEnvClient[ActT, ObsT, StateT]":
        """
        Establish connection to the server.

        Returns:
            self for method chaining
        """
        run_async_safely(self._async.connect())
        return self

    def disconnect(self) -> None:
        """Close the connection."""
        run_async_safely(self._async.disconnect())

    def reset(self, **kwargs: Any) -> StepResult[ObsT]:
        """
        Reset the environment.

        Args:
            **kwargs: Optional parameters passed to the environment's reset method

        Returns:
            StepResult containing initial observation
        """
        return run_async_safely(self._async.reset(**kwargs))

    def step(self, action: ActT, **kwargs: Any) -> StepResult[ObsT]:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute
            **kwargs: Optional parameters

        Returns:
            StepResult containing observation, reward, and done status
        """
        return run_async_safely(self._async.step(action, **kwargs))

    def state(self) -> StateT:
        """
        Get the current environment state.

        Returns:
            State object with environment state information
        """
        return run_async_safely(self._async.state())

    def close(self) -> None:
        """Close the connection and clean up resources."""
        run_async_safely(self._async.close())

    def __enter__(self) -> "SyncEnvClient[ActT, ObsT, StateT]":
        """Enter context manager, establishing connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, closing connection."""
        self.close()

    # Delegate abstract method implementations to the wrapped client
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        """Delegate to async client's _step_payload."""
        return self._async._step_payload(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        """Delegate to async client's _parse_result."""
        return self._async._parse_result(payload)

    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        """Delegate to async client's _parse_state."""
        return self._async._parse_state(payload)
