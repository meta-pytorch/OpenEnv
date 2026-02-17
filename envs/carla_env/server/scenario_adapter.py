# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapter to bridge sinatras/carla-env scenarios with OpenEnv CarlaEnvironment.

The sinatras scenarios use a different interface (state: Dict, VF-style)
while CarlaEnvironment uses simpler methods (setup, check_termination, etc.).

This adapter translates between the two.
"""

from typing import Dict, Any, Optional, List

try:
    import carla
except ImportError:
    carla = None

from .scenarios import BaseScenario as OpenEnvBaseScenario
from .benchmark_scenarios import base as benchmark_base


class SinatrasScenarioAdapter(OpenEnvBaseScenario):
    """
    Adapter that wraps sinatras BaseScenario to work with OpenEnv CarlaEnvironment.

    Translates between:
    - Sinatras: reset(state), setup(state), is_done(state), compute_outcome(state)
    - OpenEnv: setup() -> Dict, check_termination(state), compute_reward(state, action)
    """

    def __init__(self, sinatras_scenario: benchmark_base.BaseScenario, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with sinatras scenario.

        Args:
            sinatras_scenario: Instance of sinatras BaseScenario
            config: Additional configuration (optional)
        """
        super().__init__(config)
        self.sinatras_scenario = sinatras_scenario
        self.name = sinatras_scenario.__class__.__name__

        # Internal state that sinatras scenarios expect
        self._sinatras_state: Dict[str, Any] = {}

        # Track tool calls for classify_trolley_action
        self._tool_calls: List[Dict[str, Any]] = []

    def setup(self) -> Dict[str, Any]:
        """
        Setup scenario using sinatras interface.

        Translates to sinatras reset() + setup() calls.

        Returns:
            Setup dict compatible with CarlaEnvironment
        """
        # Reset sinatras state
        self._sinatras_state = {
            "scenario_state": {},
            "tool_calls": [],
            "last_action": {},
        }

        # Call sinatras reset (initializes scenario state)
        self.sinatras_scenario.reset(self._sinatras_state)

        # Extract config from sinatras scenario
        config = self.sinatras_scenario.config

        # Return setup dict for CarlaEnvironment
        return {
            "spawn_point": {
                "location": None,  # Use CARLA's spawn points (on road)
                "rotation": (0.0, 0.0, 0.0)
            },
            "initial_speed_kmh": getattr(config, 'initial_speed_kmh', 40.0),
            "weather": getattr(config, 'weather', 'ClearNoon'),
            "max_steps": getattr(config, 'max_steps', 50),
            "actors": [],  # Will be spawned by sinatras setup()
        }

    def setup_carla(self, carla_runtime: Any) -> None:
        """
        Setup CARLA actors using sinatras interface.

        This is called after CARLA world/vehicle are initialized.

        Args:
            carla_runtime: Object with .world, .ego_vehicle, .client
        """
        # Create state dict that sinatras expects
        self._sinatras_state["carla"] = carla_runtime

        # Call sinatras setup (spawns actors)
        try:
            self.sinatras_scenario.setup(self._sinatras_state)
        except Exception as e:
            # If setup fails, log but continue (e.g., spawn_requirements not met)
            print(f"Warning: Sinatras scenario setup failed: {e}")

    def check_termination(self, state: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if episode should terminate.

        Translates OpenEnv state to sinatras state format.

        Args:
            state: OpenEnv state dict

        Returns:
            (done, reason) tuple
        """
        # Update sinatras state with current CarlaEnvironment state
        self._sinatras_state.update({
            "step_count": state.get("step_count", 0),
            "speed_kmh": state.get("speed_kmh", 0.0),
            "collision_detected": state.get("collision_detected", False),
            "distance_traveled": state.get("distance_traveled", 0.0),
        })

        # Call sinatras is_done
        done = self.sinatras_scenario.is_done(self._sinatras_state)

        # Determine reason (sinatras doesn't return reason, infer it)
        reason = ""
        if done:
            if state.get("collision_detected"):
                reason = "collision"
            elif state.get("step_count", 0) >= getattr(self.sinatras_scenario.config, 'max_steps', 50):
                reason = "timeout"
            else:
                reason = "scenario_complete"

        return done, reason

    def compute_reward(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Compute reward using sinatras outcome.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Reward value
        """
        # Track tool call for classify_trolley_action
        tool_call = {
            "name": action.get("action_type", "observe"),
            "args": {
                "direction": action.get("lane_direction"),
                "steer": action.get("steer", 0.0),
                "throttle": action.get("throttle", 0.0),
                "brake": action.get("brake", 0.0),
            }
        }
        self._tool_calls.append(tool_call)
        self._sinatras_state["tool_calls"] = self._tool_calls

        # Update sinatras state
        self._sinatras_state.update(state)

        # Compute outcome
        try:
            outcome = self.sinatras_scenario.compute_outcome(self._sinatras_state)
            # Extract reward from outcome (sinatras uses 0.0-1.0 scale)
            return outcome.get("reward", 0.0)
        except Exception as e:
            print(f"Warning: compute_outcome failed: {e}")
            return 0.0

    def get_scene_description(self, state: Dict[str, Any]) -> str:
        """
        Generate scene description.

        Uses sinatras build_system_prompt if available, otherwise generic.

        Args:
            state: Current state

        Returns:
            Scene description
        """
        # Try to use sinatras build_system_prompt
        try:
            return self.sinatras_scenario.build_system_prompt(self._sinatras_state)
        except:
            # Fallback to generic description
            return f"Scenario: {self.name}\nStep: {state.get('step_count', 0)}"


def create_trolley_micro_scenario(benchmark_id: str = "classic_3v1", deadzone: bool = False) -> SinatrasScenarioAdapter:
    """
    Create TrolleyMicroScenario wrapped in adapter.

    Args:
        benchmark_id: Benchmark identifier (e.g., "classic_3v1")
        deadzone: Whether to use deadzone variant

    Returns:
        Adapted scenario compatible with CarlaEnvironment
    """
    from .benchmark_scenarios.trolley_micro import TrolleyMicroScenario, TrolleyMicroConfig

    config = TrolleyMicroConfig(
        name=f"trolley_micro_{benchmark_id}",
        description=f"Trolley micro-benchmark: {benchmark_id}",
        benchmark_id=benchmark_id,
        deadzone=deadzone,
    )

    sinatras_scenario = TrolleyMicroScenario(config)
    return SinatrasScenarioAdapter(sinatras_scenario)


def create_action_bias_scenario(center_count: int = 3, side_count: int = 1, deadzone: bool = False) -> SinatrasScenarioAdapter:
    """
    Create ActionBiasScenario wrapped in adapter.

    Args:
        center_count: Pedestrians in ego lane
        side_count: Pedestrians in side lanes
        deadzone: Whether to use deadzone variant

    Returns:
        Adapted scenario compatible with CarlaEnvironment
    """
    from .benchmark_scenarios.action_bias import ActionBiasScenario, ActionBiasConfig

    config = ActionBiasConfig(
        name=f"action_bias_{center_count}v{side_count}",
        description=f"Action bias: {center_count} center vs {side_count} side",
        center_count=center_count,
        side_count=side_count,
        deadzone=deadzone,
    )

    sinatras_scenario = ActionBiasScenario(config)
    return SinatrasScenarioAdapter(sinatras_scenario)
