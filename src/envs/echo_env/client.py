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

from typing import Dict

try:
    from core.client_types import StepResult
    from core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
        ListToolsObservation,
        Observation,
        State,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from core.http_env_client import HTTPEnvClient
except ImportError:
    from openenv_core.client_types import StepResult
    from openenv_core.env_server.mcp_types import (
        CallToolAction,
        CallToolObservation,
        ListToolsObservation,
        Observation,
        State,
        Tool,
        ToolError,
        ToolErrorType,
    )
    from openenv_core.http_env_client import HTTPEnvClient


class EchoEnv(HTTPEnvClient[CallToolAction, Observation]):
    """
    HTTP client for the Echo Environment (MCP-based).

    This client connects to an EchoEnvironment HTTP server and provides
    methods to interact with it using MCP actions.

    Example:
        >>> from core.env_server.mcp_types import CallToolAction
        >>> # Connect to a running server
        >>> client = EchoEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>>
        >>> # Call echo_message tool using step API
        >>> action = CallToolAction(tool_name="echo_message", parameters={"message": "Hello!"})
        >>> result = client.step(action)
        >>> print(result.observation.result)  # {"echoed_message": "Hello!"}

    Example with Docker:
        >>> from core.env_server.mcp_types import CallToolAction
        >>> # Automatically start container and connect
        >>> client = EchoEnv.from_docker_image("echo-env:latest")
        >>> result = client.reset()
        >>> action = CallToolAction(tool_name="echo_message", parameters={"message": "Test"})
        >>> result = client.step(action)
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
            # Parse tools into Tool objects
            tools = []
            for tool_data in obs_data.get("tools", []):
                if isinstance(tool_data, dict):
                    tools.append(
                        Tool(
                            name=tool_data.get("name", ""),
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("input_schema", {}),
                            output_schema=tool_data.get("output_schema"),
                        )
                    )
                elif isinstance(tool_data, Tool):
                    tools.append(tool_data)

            observation = ListToolsObservation(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
                tools=tools,
            )
        elif "result" in obs_data or "error" in obs_data or "tool_name" in obs_data:
            # Parse error into ToolError if present
            error = None
            error_data = obs_data.get("error")
            if error_data:
                if isinstance(error_data, dict):
                    error_type_str = error_data.get("error_type", "internal_error")
                    try:
                        error_type = ToolErrorType(error_type_str)
                    except ValueError:
                        error_type = ToolErrorType.INTERNAL_ERROR
                    error = ToolError(
                        error_type=error_type,
                        message=error_data.get("message", str(error_data)),
                        details=error_data.get("details"),
                    )
                elif isinstance(error_data, ToolError):
                    error = error_data

            observation = CallToolObservation(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
                result=obs_data.get("result"),
                error=error,
                tool_name=obs_data.get("tool_name", ""),
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
