# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure and demonstrating
how to build an MCPEnvironment subclass.

Supports both traditional EchoAction and MCP actions (ListToolsAction, CallToolAction)
for backwards compatibility while enabling MCP tool discovery and invocation.
"""

from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
    from ..models import EchoAction, EchoObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
    from models import EchoAction, EchoObservation

from .mcp_server import mcp


class EchoEnvironment(MCPEnvironment):
    """
    A simple echo environment that echoes back messages.

    This environment demonstrates how to build an MCPEnvironment subclass.
    It inherits MCP support (ListToolsAction, CallToolAction) from the base class
    and implements _step_impl() for handling domain-specific actions (EchoAction).

    Example:
        >>> env = EchoEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Echo environment ready!"
        >>>
        >>> # Traditional action
        >>> obs = env.step(EchoAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>>
        >>> # MCP action - list tools
        >>> from openenv.core.env_server.mcp_types import ListToolsAction
        >>> obs = env.step(ListToolsAction())
        >>> print([t.name for t in obs.tools])  # ["echo_message", "echo_with_length"]
        >>>
        >>> # MCP action - call tool
        >>> from openenv.core.env_server.mcp_types import CallToolAction
        >>> obs = env.step(CallToolAction(tool_name="echo_message", arguments={"message": "Hi!"}))
        >>> print(obs.result)  # "Hi!"
    """

    def __init__(self):
        """Initialize the echo environment with MCP server."""
        # Pass the MCP server to the base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EchoObservation:
        """
        Reset the environment.

        Args:
            seed: Optional random seed (unused in echo env)
            episode_id: Optional episode ID to use
            **kwargs: Additional reset options

        Returns:
            EchoObservation with a ready message
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        return EchoObservation(
            echoed_message="Echo environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions (EchoAction).

        MCP actions (ListToolsAction, CallToolAction) are handled by the base class.
        This method handles the domain-specific EchoAction.

        Args:
            action: The action to execute (should be EchoAction)
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            EchoObservation with echoed message
        """
        # Note: step count is incremented in step() before calling this

        # Handle EchoAction
        if isinstance(action, EchoAction):
            message = action.message
            length = len(message)

            # Simple reward: longer messages get higher rewards
            reward = length * 0.1

            return EchoObservation(
                echoed_message=message,
                message_length=length,
                done=False,
                reward=reward,
                metadata={"original_message": message, "step_count": self._state.step_count},
            )

        # Unknown action type
        return EchoObservation(
            echoed_message="Unknown action type",
            message_length=0,
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}"},
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions, handles EchoAction locally.
        Also increments step count for all actions.
        """
        # Increment step count for all actions (including MCP)
        self._state.step_count += 1

        # Let the base class handle MCP actions and non-MCP routing
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
