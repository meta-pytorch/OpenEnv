"""HTTP client for cognitive manufacturing environment.

This module provides a convenient wrapper for interacting with the
manufacturing environment server via HTTP.
"""

from __future__ import annotations
from typing import Any
from dataclasses import asdict
from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult
from .models import ManufacturingAction, ManufacturingObservation


class CognitiveManufacturingEnv(HTTPEnvClient[ManufacturingAction, ManufacturingObservation]):
    """HTTP client for cognitive manufacturing environment.

    This client connects to a running manufacturing environment server
    and provides a simple interface for agents to interact with it.

    Usage:
        >>> env = CognitiveManufacturingEnv(base_url="http://localhost:8000")
        >>> result = env.reset()
        >>> obs = result.observation
        >>> action = ManufacturingAction(
        ...     tool_name="ReadSensors",
        ...     parameters={"machine_id": "M1", "sensors": "all"}
        ... )
        >>> result = env.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client.

        Args:
            base_url: Base URL of the environment server
        """
        super().__init__(base_url=base_url)

    def _step_payload(self, action: ManufacturingAction) -> dict:
        """Convert ManufacturingAction to JSON payload."""
        return asdict(action)

    def _parse_result(self, payload: dict) -> StepResult[ManufacturingObservation]:
        """Parse server response into StepResult."""
        # The server returns: {observation, reward, done, info}
        obs_data = payload.get("observation", {})

        # Reconstruct ManufacturingObservation
        # Note: This is simplified - in production you'd properly reconstruct the dataclass
        observation = ManufacturingObservation(
            tool_result=obs_data.get("tool_result", {}),
            machine_status=obs_data.get("machine_status"),  # Dict format is fine
            alerts=obs_data.get("alerts", []),
            simulation_time=obs_data.get("simulation_time", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            info=payload.get("info", {}),
        )

    def _parse_state(self, payload: dict) -> Any:
        """Parse state endpoint response."""
        return payload

    def get_available_tools(self) -> list[str]:
        """Get list of available tools from the environment.

        Returns:
            List of tool names
        """
        state = self.state()
        return state.get("available_tools", [])

    def get_machine_status(self) -> dict[str, Any]:
        """Get current machine status.

        Returns:
            Dictionary with machine status information
        """
        state = self.state()
        return state.get("machine", {})

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get recent alerts.

        Returns:
            List of recent alert dictionaries
        """
        state = self.state()
        return state.get("alerts", [])
