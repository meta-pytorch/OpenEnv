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
        action_type: Type of action (control, emergency_stop, lane_change, observe, maintain_speed, brake_vehicle)
        throttle: Throttle value [0.0, 1.0] for "control" actions
        steer: Steering value [-1.0, 1.0] for "control" actions
        brake: Brake value [0.0, 1.0] for "control" actions
        lane_direction: Direction for "lane_change" ("left" or "right")
        target_speed_kmh: Target speed in km/h for "maintain_speed"
        brake_intensity: Brake intensity [0.0, 1.0] for "brake_vehicle" (NEW - Day 2)
        target_lane_id: Target lane ID for improved "lane_change" (NEW - Day 2)
    """
    action_type: str = Field(default="observe", description="Type of action")
    throttle: float = Field(default=0.0, ge=0.0, le=1.0, description="Throttle value")
    steer: float = Field(default=0.0, ge=-1.0, le=1.0, description="Steering value")
    brake: float = Field(default=0.0, ge=0.0, le=1.0, description="Brake value")
    lane_direction: Optional[str] = Field(default=None, description="Lane change direction (deprecated, use target_lane_id)")

    # Day 2: New action parameters
    target_speed_kmh: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Target speed in km/h for maintain_speed action"
    )
    brake_intensity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Brake intensity (0.0 = no brake, 1.0 = full brake) for brake_vehicle action"
    )
    target_lane_id: Optional[str] = Field(
        default=None,
        description="Target lane ID for lane_change action (e.g., 'lane_0', 'lane_1')"
    )


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

    # Day 3: Action tracking metrics (from sinatras)
    num_turns: int = Field(default=0, description="Number of steps taken in episode")
    total_tool_calls: int = Field(default=0, description="Total number of actions executed")
    tool_call_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of each action type executed"
    )
    is_truncated: bool = Field(default=False, description="Whether episode was truncated (max steps)")

    # Day 3: Movement metrics
    average_speed: float = Field(default=0.0, description="Average speed in km/h")
    max_speed: float = Field(default=0.0, description="Maximum speed reached in km/h")

    # Collision history
    collisions: List[Dict[str, Any]] = Field(default_factory=list, description="List of collision events")
    collisions_count: int = Field(default=0, description="Total number of collisions")
    collision_intensity_total: float = Field(
        default=0.0,
        description="Sum of all collision intensities"
    )

    # Scenario-specific data
    scenario_data: Dict[str, Any] = Field(default_factory=dict, description="Scenario-specific data")
