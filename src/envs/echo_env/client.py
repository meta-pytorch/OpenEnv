# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment HTTP Client (MCP-based).

This module provides the client for connecting to an Echo Environment server
over HTTP using MCP actions.
"""

try:
    from core.client_types import StepResult
    from core.env_server.types import (
        CallToolAction,
        CallToolObservation,
        ListToolsObservation,
        Observation,
        State,
    )
    from core.http_env_client import HTTPEnvClient
except ImportError:
    from openenv_core.client_types import StepResult
    from openenv_core.env_server.types import (
        CallToolAction,
        CallToolObservation,
        ListToolsObservation,
        Observation,
        State,
    )
    from openenv_core.http_env_client import HTTPEnvClient

from typing import Any, Dict, List


class EchoEnv(HTTPEnvClient[CallToolAction, Observation]):
    """
    HTTP client for the Echo Environment (MCP-based).

    This client connects to an EchoEnvironment HTTP server and provides
    methods to interact with it using MCP actions.

    Example:
        >>> # Connect to a running server
        >>> client = EchoEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>>
        >>> # List available tools
        >>> tools = client.list_tools()
        >>> print(tools)  # [{"name": "echo_message", ...}]
        >>>
        >>> # Call echo_message tool
        >>> result = client.echo_message("Hello!")
        >>> print(result["echoed_message"])  # "Hello!"

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = EchoEnv.from_docker_image("echo-env:latest")
        >>> result = client.reset()
        >>> result = client.echo_message("Test")
    """

    def _step_payload(self, action: CallToolAction) -> Dict:
        """
        Convert CallToolAction to JSON payload for step request.

        Args:
            action: CallToolAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "type": "CallToolAction",
            "tool_name": action.tool_name,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """
        Parse server response into StepResult with typed Observation.

        Args:
            payload: JSON response from server

        Returns:
            StepResult with typed Observation (ListToolsObservation or CallToolObservation)
        """
        obs_data = payload.get("observation", {})

        # Create appropriate typed observation based on fields present
        if "tools" in obs_data:
            observation = ListToolsObservation(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
                tools=obs_data.get("tools", []),
            )
        elif "result" in obs_data or "error" in obs_data or "tool_name" in obs_data:
            observation = CallToolObservation(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
                result=obs_data.get("result"),
                error=obs_data.get("error"),
                tool_name=obs_data.get("tool_name"),
            )
        else:
            observation = Observation(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns:
            List of tool dictionaries with name, description, and inputSchema
        """
        # Send action via step with correct payload format
        payload = {"type": "ListToolsAction"}
        response = self._session.post(f"{self.base_url}/step", json={"action": payload})
        response.raise_for_status()
        result = self._parse_result(response.json())
        if isinstance(result.observation, ListToolsObservation):
            return result.observation.tools
        return []

    def echo_message(self, message: str) -> Dict[str, Any]:
        """
        Call the echo_message tool.

        Args:
            message: The message to echo

        Returns:
            Dictionary containing the echoed message result
        """
        action = CallToolAction(tool_name="echo_message", parameters={"message": message})
        result = self.step(action)
        if isinstance(result.observation, CallToolObservation):
            return result.observation.result or {}
        return {}
