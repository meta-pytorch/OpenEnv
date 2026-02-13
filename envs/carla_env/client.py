# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for CARLA environment.

Provides EnvClient wrapper for remote or local CARLA instances.
"""

from typing import Optional, Dict, Any
from openenv.core.env_client import EnvClient, StepResult
from .models import CarlaAction, CarlaObservation, CarlaState


class CarlaEnv(EnvClient[CarlaAction, CarlaObservation, CarlaState]):
    """
    Client for CARLA environment.

    Connects to a running CARLA environment server via WebSocket.

    Example:
        >>> from carla_env import CarlaEnv, CarlaAction
        >>> env = CarlaEnv(base_url="http://localhost:8000")
        >>> result = env.reset()
        >>> print(result.observation.scene_description)
        >>> result = env.step(CarlaAction(action_type="emergency_stop"))
        >>> env.close()

    For async usage:
        >>> async with CarlaEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     result = await env.step(CarlaAction(action_type="observe"))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        **kwargs
    ):
        """
        Initialize CARLA environment client.

        Args:
            base_url: Base URL of the CARLA environment server
            **kwargs: Additional arguments for EnvClient
        """
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: CarlaAction) -> Dict[str, Any]:
        """Convert CarlaAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CarlaObservation]:
        """Parse JSON response to StepResult."""
        observation = CarlaObservation(**payload["observation"])
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=observation.done
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CarlaState:
        """Parse JSON response to CarlaState."""
        return CarlaState(**payload)

    @classmethod
    def from_docker_image(
        cls,
        image: str = "carla-env:latest",
        scenario: str = "trolley_saves",
        mode: str = "mock",
        **kwargs
    ) -> "CarlaEnv":
        """
        Create CARLA environment from Docker image.

        Args:
            image: Docker image name
            scenario: Scenario to run
            mode: "mock" or "real"
            **kwargs: Additional Docker run arguments

        Returns:
            CarlaEnv instance connected to container
        """
        from openenv.core.containers import LocalDockerProvider

        provider = LocalDockerProvider()

        # Environment variables for configuration
        environment = {
            "CARLA_SCENARIO": scenario,
            "CARLA_MODE": mode,
        }

        if "environment" in kwargs:
            environment.update(kwargs.pop("environment"))

        container = provider.create_container(
            image=image,
            environment=environment,
            **kwargs
        )

        provider.start_container(container.id)

        # Get container URL
        base_url = f"http://localhost:{container.ports.get('8000', 8000)}"

        return cls(base_url=base_url)
