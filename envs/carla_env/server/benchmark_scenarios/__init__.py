# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CARLA Scenarios adapted from sinatras/carla-env.

Scenarios for evaluating LLM decision-making in autonomous driving contexts.
"""

from typing import Optional, Dict, Any

from .base import BaseScenario, ScenarioConfig
from .shared import TrolleyAction, classify_trolley_action, same_direction
from .trolley_micro import TrolleyMicroConfig, TrolleyMicroScenario
from .action_bias import ActionBiasConfig, ActionBiasScenario

# Import simple scenarios from parent module (scenarios.py file, not this directory)
try:
    from .. import scenarios as scenarios_module
    SimpleTrolleyScenario = scenarios_module.SimpleTrolleyScenario
    MazeNavigationScenario = scenarios_module.MazeNavigationScenario
    SIMPLE_SCENARIOS = scenarios_module.SCENARIOS
except (ImportError, AttributeError):
    # Fallback: define minimal versions here
    SimpleTrolleyScenario = None
    MazeNavigationScenario = None
    SIMPLE_SCENARIOS = {}


def get_scenario(scenario_name: str, config: Optional[Dict[str, Any]] = None) -> BaseScenario:
    """
    Get scenario by name.

    Supports both simple scenarios and sinatras/carla-env scenarios:
    - trolley_saves, trolley_equal - Simple trolley scenarios
    - maze_navigation - Simple maze scenario
    - trolley_micro_<benchmark_id> - Trolley micro-benchmarks
    - trolley_micro_<benchmark_id>_deadzone - With deadzone
    - action_bias_saves, action_bias_less, action_bias_equal - Action-bias variants
    - bias_<N>v<M> - Custom action-bias (N center, M side)
    - bias_<N>v<M>_deadzone - Custom with deadzone

    Args:
        scenario_name: Name of scenario
        config: Optional configuration override

    Returns:
        Scenario instance
    """
    # Check simple scenarios first
    if scenario_name in SIMPLE_SCENARIOS:
        scenario = SIMPLE_SCENARIOS[scenario_name]()
        if config:
            scenario.config.update(config)
        return scenario

    # Parse sinatras scenarios
    from ..scenario_adapter import create_trolley_micro_scenario, create_action_bias_scenario

    # Trolley micro-benchmarks
    if scenario_name.startswith("trolley_micro_"):
        # Remove prefix
        rest = scenario_name[len("trolley_micro_"):]

        # Check for deadzone
        deadzone = False
        if rest.endswith("_deadzone"):
            deadzone = True
            rest = rest[:-len("_deadzone")]

        benchmark_id = rest
        return create_trolley_micro_scenario(benchmark_id=benchmark_id, deadzone=deadzone)

    # Action-bias scenarios
    if scenario_name.startswith("action_bias_"):
        variant = scenario_name[len("action_bias_"):]

        if variant == "saves":
            return create_action_bias_scenario(center_count=5, side_count=0)
        elif variant == "less":
            return create_action_bias_scenario(center_count=3, side_count=1)
        elif variant == "equal":
            return create_action_bias_scenario(center_count=2, side_count=2)
        else:
            raise ValueError(f"Unknown action_bias variant: {variant}")

    # Custom bias scenarios: bias_<N>v<M> or bias_<N>v<M>_deadzone
    if scenario_name.startswith("bias_"):
        # Remove prefix
        rest = scenario_name[len("bias_"):]

        # Check for deadzone
        deadzone = False
        if rest.endswith("_deadzone"):
            deadzone = True
            rest = rest[:-len("_deadzone")]

        # Parse NvM format
        try:
            parts = rest.split("v")
            if len(parts) != 2:
                raise ValueError()
            center_count = int(parts[0])
            side_count = int(parts[1])
            return create_action_bias_scenario(
                center_count=center_count,
                side_count=side_count,
                deadzone=deadzone
            )
        except:
            raise ValueError(f"Invalid bias format: {scenario_name}. Use bias_<N>v<M> (e.g., bias_3v1)")

    raise ValueError(f"Unknown scenario: {scenario_name}")


__all__ = [
    "BaseScenario",
    "ScenarioConfig",
    "TrolleyAction",
    "classify_trolley_action",
    "same_direction",
    "TrolleyMicroScenario",
    "TrolleyMicroConfig",
    "ActionBiasScenario",
    "ActionBiasConfig",
    "SimpleTrolleyScenario",
    "MazeNavigationScenario",
    "get_scenario",
]
