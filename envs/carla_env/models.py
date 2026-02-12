# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for CARLA environment.

Defines Action, Observation, and State for embodied evaluation scenarios.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class CarlaAction(Action):
    """
    Action for CARLA vehicle control.

    Attributes:
        action_type: Type of action ("control", "emergency_stop", "lane_change", "observe")
        throttle: Throttle value [0.0, 1.0] for "control" actions
        steer: Steering value [-1.0, 1.0] for "control" actions
        brake: Brake value [0.0, 1.0] for "control" actions
        lane_direction: Direction for "lane_change" ("left" or "right")
    """
    action_type: str = Field(default="observe", description="Type of action")
    throttle: float = Field(default=0.0, ge=0.0, le=1.0, description="Throttle value")
    steer: float = Field(default=0.0, ge=-1.0, le=1.0, description="Steering value")
    brake: float = Field(default=0.0, ge=0.0, le=1.0, description="Brake value")
    lane_direction: Optional[str] = Field(default=None, description="Lane change direction")


class CarlaObservation(Observation):
    """
    Observation from CARLA environment.

    For text-only mode, provides ground truth scene description.
    """
    # Scene description (text-only mode)
    scene_description: str = Field(default="", description="Natural language scene description")

    # Vehicle state
    speed_kmh: float = Field(default=0.0, description="Current speed in km/h")
    location: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Vehicle location (x, y, z)")
    rotation: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Vehicle rotation (pitch, yaw, roll)")

    # Lane info
    current_lane: str = Field(default="unknown", description="Current lane identifier")

    # Nearby actors (for decision-making)
    nearby_actors: List[Dict[str, Any]] = Field(default_factory=list, description="Nearby actors with distances")

    # Collision detection
    collision_detected: bool = Field(default=False, description="Whether collision occurred")
    collision_intensity: float = Field(default=0.0, description="Collision force intensity")
    collided_with: Optional[str] = Field(default=None, description="ID of actor collided with")

    # Scenario info
    scenario_name: str = Field(default="", description="Name of current scenario")
    simulation_time: float = Field(default=0.0, description="Simulation time in seconds")
    step_number: int = Field(default=0, description="Current step number")

    # Episode termination (override done from base Observation)
    done_reason: str = Field(default="", description="Reason for episode termination")


class CarlaState(State):
    """
    Episode state for CARLA environment.
    """
    # Scenario configuration
    scenario_name: str = Field(default="default", description="Name of current scenario")
    town: str = Field(default="Town10HD_Opt", description="CARLA town/map name")
    weather: str = Field(default="ClearNoon", description="Weather preset")

    # Episode metrics
    total_distance: float = Field(default=0.0, description="Total distance traveled (meters)")
    total_reward: float = Field(default=0.0, description="Cumulative reward")
    simulation_time: float = Field(default=0.0, description="Total simulation time (seconds)")

    # Collision history
    collisions: List[Dict[str, Any]] = Field(default_factory=list, description="List of collision events")

    # Scenario-specific data
    scenario_data: Dict[str, Any] = Field(default_factory=dict, description="Scenario-specific data")
