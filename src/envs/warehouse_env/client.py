"""
HTTP client for Warehouse Optimization Environment.

This module provides the client-side interface for interacting with
the warehouse environment server.
"""

from typing import Optional

import requests

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient
from envs.warehouse_env.models import (
    WarehouseAction,
    WarehouseObservation,
    WarehouseState,
)


class WarehouseEnv(HTTPEnvClient[WarehouseAction, WarehouseObservation]):
    """
    HTTP client for the Warehouse Optimization environment.

    This environment simulates a warehouse robot that must pick up and deliver
    packages while navigating a grid-based warehouse with obstacles.

    Example usage:
        ```python
        from envs.warehouse_env import WarehouseEnv, WarehouseAction

        # Start environment from Docker image
        env = WarehouseEnv.from_docker_image(
            "warehouse-env:latest",
            environment={"DIFFICULTY_LEVEL": "2"}
        )

        # Reset environment
        result = env.reset()
        print(f"Grid size: {len(result.observation.grid)}x{len(result.observation.grid[0])}")
        print(f"Packages: {result.observation.total_packages}")

        # Take actions
        for step in range(100):
            # Example: move randomly or pick/drop based on state
            if result.observation.robot_carrying is None:
                action = WarehouseAction(action_id=4)  # Try to pick up
            else:
                action = WarehouseAction(action_id=5)  # Try to drop off

            result = env.step(action)
            print(f"Step {step}: {result.observation.message}")

            if result.done:
                print(f"Episode finished! Delivered {result.observation.packages_delivered} packages")
                break

        env.close()
        ```

    Configuration (via environment variables):
        - DIFFICULTY_LEVEL: 1-5 (default: 2)
          1 = Simple (5x5, 1 package)
          2 = Easy (8x8, 2 packages)
          3 = Medium (10x10, 3 packages)
          4 = Hard (15x15, 5 packages)
          5 = Expert (20x20, 8 packages)

        - GRID_WIDTH: Custom grid width (overrides difficulty)
        - GRID_HEIGHT: Custom grid height (overrides difficulty)
        - NUM_PACKAGES: Custom number of packages (overrides difficulty)
        - MAX_STEPS: Maximum steps per episode (default: based on difficulty)
        - RANDOM_SEED: Random seed for reproducibility (default: None)
    """

    def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
        """Initialize warehouse environment client."""
        super().__init__(base_url=base_url, **kwargs)
        self.base_url = base_url  # Store for render_ascii() method

    def _step_payload(self, action: WarehouseAction) -> dict:
        """Convert WarehouseAction to JSON payload for step request."""
        return {"action_id": action.action_id}

    def _parse_result(self, payload: dict) -> StepResult[WarehouseObservation]:
        """Parse server response into StepResult[WarehouseObservation]."""
        obs_data = payload.get("observation", {})

        observation = WarehouseObservation(
            grid=obs_data.get("grid", []),
            robot_position=tuple(obs_data.get("robot_position", [0, 0])),
            robot_carrying=obs_data.get("robot_carrying"),
            packages=obs_data.get("packages", []),
            step_count=obs_data.get("step_count", 0),
            packages_delivered=obs_data.get("packages_delivered", 0),
            total_packages=obs_data.get("total_packages", 0),
            time_remaining=obs_data.get("time_remaining", 0),
            action_success=obs_data.get("action_success", False),
            message=obs_data.get("message", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WarehouseState:
        """Parse server response into WarehouseState object."""
        return WarehouseState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            packages_delivered=payload.get("packages_delivered", 0),
            total_packages=payload.get("total_packages", 0),
            difficulty_level=payload.get("difficulty_level", 2),
            grid_size=tuple(payload.get("grid_size", [0, 0])),
            cum_reward=payload.get("cum_reward", 0.0),
            is_done=payload.get("is_done", False),
        )

    def render_ascii(self) -> str:
        """
        Get ASCII visualization of the current warehouse state.

        Returns:
            String representation of the warehouse grid
        """
        try:
            response = requests.get(f"{self.base_url}/render")
            response.raise_for_status()
            return response.json()["ascii"]
        except Exception as e:
            return f"Error rendering: {str(e)}"
