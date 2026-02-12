# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CARLA Environment implementation for OpenEnv.

Supports two modes:
1. Real mode: Connects to CARLA server (requires carla package)
2. Mock mode: Simulated physics for testing without CARLA

The environment wraps CARLA scenarios and provides OpenEnv-compatible API.
"""

import uuid
import math
from typing import Optional, Dict, Any
from openenv.core.env_server import Environment

from ..models import CarlaAction, CarlaObservation, CarlaState
from .scenarios import BaseScenario, get_scenario

# Try to import CARLA, but don't fail if not available
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False
    carla = None


class CarlaEnvironment(Environment):
    """
    CARLA environment for embodied evaluation.

    Supports scenario-based testing where:
    - Time flows continuously (simulation clock)
    - Actions have irreversible consequences
    - Inaction is itself a measurable choice

    Args:
        scenario_name: Name of scenario to run
        host: CARLA server host (for real mode)
        port: CARLA server port (for real mode)
        mode: "real" (requires CARLA) or "mock" (simulated)
        scenario_config: Optional scenario configuration
    """

    def __init__(
        self,
        scenario_name: str = "trolley_saves",
        host: str = "localhost",
        port: int = 2000,
        mode: str = "mock",
        scenario_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Mode selection
        self.mode = mode
        if self.mode == "real" and not CARLA_AVAILABLE:
            raise ImportError(
                "CARLA package not available. Install with: pip install carla\n"
                "Or use mode='mock' for simulated physics."
            )

        # Connection params
        self.host = host
        self.port = port

        # Load scenario
        self.scenario: BaseScenario = get_scenario(scenario_name, scenario_config)

        # State
        self._state = CarlaState(scenario_name=scenario_name)

        # CARLA connection (real mode only)
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # Mock mode state
        self.mock_state: Dict[str, Any] = {}

        # Scenario data
        self.scenario_data: Dict[str, Any] = {}

    def reset(self) -> CarlaObservation:
        """
        Reset environment and setup scenario.

        Returns:
            Initial observation
        """
        # Generate new episode ID
        self._state = CarlaState(
            episode_id=str(uuid.uuid4()),
            scenario_name=self.scenario.name,
            step_count=0,
        )

        # Setup scenario
        setup = self.scenario.setup()
        self.scenario_data = setup

        # Initialize based on mode
        if self.mode == "real":
            self._reset_real_mode(setup)
        else:
            self._reset_mock_mode(setup)

        # Get initial observation
        return self._get_observation()

    def step(self, action: CarlaAction) -> CarlaObservation:
        """
        Execute action and advance simulation.

        Auto-resets if environment not initialized (handles distributed deployment
        edge cases where state may not persist between HTTP requests).

        In real mode: Apply control to CARLA vehicle and tick world
        In mock mode: Update simulated physics

        Args:
            action: Action to execute

        Returns:
            Observation after action
        """
        # Ensure environment has been reset (auto-reset if needed)
        if self.mode == "real" and (self.world is None or self.vehicle is None):
            # Auto-reset on first step
            self.reset()

        # Increment step counter
        self._state.step_count += 1

        # Execute action
        if self.mode == "real":
            self._step_real_mode(action)
        else:
            self._step_mock_mode(action)

        # Get observation
        obs = self._get_observation()

        # Compute reward
        state_dict = self._get_state_dict()
        action_dict = {
            "action_type": action.action_type,
            "throttle": action.throttle,
            "steer": action.steer,
            "brake": action.brake,
        }
        reward = self.scenario.compute_reward(state_dict, action_dict)
        self._state.total_reward += reward

        return obs

    @property
    def state(self) -> CarlaState:
        """Get current episode state."""
        return self._state

    def _reset_real_mode(self, setup: Dict[str, Any]) -> None:
        """
        Reset in real CARLA mode.

        Implementation notes:
        - Uses get_world() instead of load_world() (world pre-loaded by CARLA)
        - Cleans up previous vehicle to prevent actor accumulation
        - Falls back to any vehicle if Tesla Model 3 blueprint not found

        Args:
            setup: Scenario configuration dict
        """
        # Connect to CARLA server
        if self.client is None:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)

        # Get current world (don't reload - CARLA is already running with a world)
        if self.world is None:
            self.world = self.client.get_world()

        # Clean up previous actors if they exist
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Set weather
        weather_name = setup.get("weather", "ClearNoon")
        weather = getattr(carla.WeatherParameters, weather_name)
        self.world.set_weather(weather)

        # Spawn vehicle
        spawn_point = setup.get("spawn_point", {})
        loc = spawn_point.get("location", (0.0, 0.0, 0.5))
        rot = spawn_point.get("rotation", (0.0, 0.0, 0.0))

        blueprint_library = self.world.get_blueprint_library()

        # Try to find Tesla Model 3, fallback to any vehicle if not available
        try:
            vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        except RuntimeError:
            # Tesla not available in this CARLA version, use any car
            vehicles = blueprint_library.filter("vehicle.*")
            vehicle_bp = vehicles[0] if vehicles else None
            if vehicle_bp is None:
                raise RuntimeError("No vehicle blueprints available in CARLA")

        transform = carla.Transform(
            carla.Location(x=loc[0], y=loc[1], z=loc[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)

        # Set initial speed
        initial_speed = setup.get("initial_speed_kmh", 0.0) / 3.6  # Convert to m/s
        if initial_speed > 0:
            forward_vec = self.vehicle.get_transform().get_forward_vector()
            self.vehicle.set_target_velocity(
                carla.Vector3D(
                    x=forward_vec.x * initial_speed,
                    y=forward_vec.y * initial_speed,
                    z=0.0,
                )
            )

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)

        # Initial tick
        self.world.tick()

    def _reset_mock_mode(self, setup: Dict[str, Any]) -> None:
        """Reset in mock simulation mode."""
        # Initialize mock state
        spawn_point = setup.get("spawn_point", {})
        loc = spawn_point.get("location", (0.0, 0.0, 0.5))
        rot = spawn_point.get("rotation", (0.0, 0.0, 0.0))

        self.mock_state = {
            "location": list(loc),
            "rotation": list(rot),
            "velocity": [0.0, 0.0, 0.0],
            "speed_kmh": setup.get("initial_speed_kmh", 0.0),
            "actors": setup.get("actors", []),
            "collisions": [],
            "time": 0.0,
            "delta_time": 0.05,  # 20 FPS
        }

    def _step_real_mode(self, action: CarlaAction) -> None:
        """Execute action in real CARLA mode."""
        if action.action_type == "control":
            control = carla.VehicleControl(
                throttle=action.throttle,
                steer=action.steer,
                brake=action.brake,
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "emergency_stop":
            control = carla.VehicleControl(brake=1.0, throttle=0.0)
            self.vehicle.apply_control(control)

        elif action.action_type == "lane_change":
            # Simplified lane change: apply lateral steering
            steer = -0.5 if action.lane_direction == "left" else 0.5
            control = carla.VehicleControl(throttle=0.3, steer=steer)
            self.vehicle.apply_control(control)

        # Tick simulation
        self.world.tick()

    def _step_mock_mode(self, action: CarlaAction) -> None:
        """Execute action in mock simulation mode."""
        dt = self.mock_state["delta_time"]

        # Apply action to mock physics
        if action.action_type == "control":
            # Update speed based on throttle/brake
            accel = action.throttle * 3.0 - action.brake * 8.0  # m/s^2
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms + accel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

            # Update position (simplified: straight line + steering)
            yaw_rad = math.radians(self.mock_state["rotation"][1])
            yaw_rad += action.steer * 0.5 * dt  # Steering effect

            dx = speed_ms * math.cos(yaw_rad) * dt
            dy = speed_ms * math.sin(yaw_rad) * dt

            self.mock_state["location"][0] += dx
            self.mock_state["location"][1] += dy
            self.mock_state["rotation"][1] = math.degrees(yaw_rad)

        elif action.action_type == "emergency_stop":
            # Strong deceleration
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms - 8.0 * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "lane_change":
            # Lateral offset (simplified)
            offset = -3.5 if action.lane_direction == "left" else 3.5
            yaw_rad = math.radians(self.mock_state["rotation"][1])
            self.mock_state["location"][0] += offset * math.sin(yaw_rad)
            self.mock_state["location"][1] += offset * math.cos(yaw_rad)

        # Check collisions (simplified)
        self._check_mock_collisions()

        # Update time
        self.mock_state["time"] += dt
        self._state.simulation_time = self.mock_state["time"]

    def _check_mock_collisions(self) -> None:
        """Check for collisions in mock mode (simplified)."""
        vehicle_pos = self.mock_state["location"]

        for actor in self.mock_state["actors"]:
            if actor["type"] == "pedestrian":
                # Compute distance to actor
                actor_distance = actor["distance"]
                actor_lateral_offset = actor.get("lane_offset", 0.0)

                # Vehicle has traveled forward
                distance_traveled = self.mock_state["speed_kmh"] / 3.6 * self.mock_state["time"]

                # Simple collision check
                if abs(distance_traveled - actor_distance) < 2.0:
                    if abs(actor_lateral_offset) < 1.5:  # Within vehicle width
                        # Collision!
                        collision = {
                            "frame": self._state.step_count,
                            "actor_id": actor["id"],
                            "intensity": self.mock_state["speed_kmh"],
                        }
                        self.mock_state["collisions"].append(collision)
                        self._state.collisions.append(collision)

    def _get_observation(self) -> CarlaObservation:
        """Generate observation from current state."""
        # Get state dict for scenario
        state_dict = self._get_state_dict()

        # Check termination
        done, done_reason = self.scenario.check_termination(state_dict)

        # Generate scene description
        scene_description = self.scenario.get_scene_description(state_dict)

        # Build observation
        if self.mode == "real":
            obs = self._get_observation_real()
        else:
            obs = self._get_observation_mock()

        obs.scene_description = scene_description
        obs.scenario_name = self.scenario.name
        obs.simulation_time = self._state.simulation_time
        obs.step_number = self._state.step_count
        obs.done = done
        obs.done_reason = done_reason

        return obs

    def _get_observation_real(self) -> CarlaObservation:
        """Get observation from real CARLA."""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        return CarlaObservation(
            speed_kmh=speed_kmh,
            location=(transform.location.x, transform.location.y, transform.location.z),
            rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            current_lane="lane_0",  # Simplified
            nearby_actors=self._get_nearby_actors_real(),
        )

    def _get_observation_mock(self) -> CarlaObservation:
        """Get observation from mock state."""
        collision_detected = len(self.mock_state["collisions"]) > 0
        collided_with = None
        if collision_detected:
            collided_with = self.mock_state["collisions"][-1]["actor_id"]

        return CarlaObservation(
            speed_kmh=self.mock_state["speed_kmh"],
            location=tuple(self.mock_state["location"]),
            rotation=tuple(self.mock_state["rotation"]),
            current_lane="lane_0",
            nearby_actors=self._get_nearby_actors_mock(),
            collision_detected=collision_detected,
            collided_with=collided_with,
        )

    def _get_nearby_actors_real(self) -> list:
        """
        Get nearby actors from CARLA world.

        NOTE: Currently returns empty list. Works for static trolley scenarios
        where pedestrians are spawned by the scenario and don't move. For
        dynamic traffic scenarios, would need to query CARLA world actors:

        Example future implementation:
            world_actors = self.world.get_actors()
            ego_location = self.vehicle.get_location()
            nearby = []
            for actor in world_actors:
                if actor.id == self.vehicle.id:
                    continue
                distance = actor.get_location().distance(ego_location)
                if distance < 50.0:
                    nearby.append({
                        "type": actor.type_id,
                        "id": actor.id,
                        "distance": distance,
                    })
            return nearby
        """
        # TODO: Implement for dynamic scenarios
        return []

    def _get_nearby_actors_mock(self) -> list:
        """Get nearby actors from mock state."""
        # Compute distance traveled
        distance_traveled = self.mock_state["speed_kmh"] / 3.6 * self.mock_state["time"]

        nearby = []
        for actor in self.mock_state["actors"]:
            # Relative distance
            relative_distance = actor["distance"] - distance_traveled

            if relative_distance > -5.0 and relative_distance < 50.0:
                nearby.append({
                    "type": actor["type"],
                    "id": actor["id"],
                    "distance": max(0.0, relative_distance),
                    "position": actor["position"],
                })

        return nearby

    def _get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dict for scenario logic."""
        if self.mode == "real":
            obs = self._get_observation_real()
        else:
            obs = self._get_observation_mock()

        return {
            "step_count": self._state.step_count,
            "speed_kmh": obs.speed_kmh,
            "location": obs.location,
            "rotation": obs.rotation,
            "current_lane": obs.current_lane,
            "nearby_actors": obs.nearby_actors,
            "collision_detected": obs.collision_detected,
            "collided_with": obs.collided_with,
            "collisions": self._state.collisions,
            "simulation_time": self._state.simulation_time,
            "distance_traveled": self.mock_state.get("speed_kmh", 0.0) / 3.6 * self._state.simulation_time,
            "done": obs.done,
            "done_reason": obs.done_reason,
            # Scenario-specific
            "goal_distance": self._compute_goal_distance(),
            "goal_direction": self._compute_goal_direction(),
            "prev_goal_distance": self.scenario_data.get("prev_goal_distance", float("inf")),
        }

    def _compute_goal_distance(self) -> float:
        """Compute distance to goal (for navigation scenarios)."""
        if "goal_location" not in self.scenario_data:
            return float("inf")

        goal = self.scenario_data["goal_location"]
        if self.mode == "real":
            loc = self.vehicle.get_transform().location
            current = (loc.x, loc.y, loc.z)
        else:
            current = self.mock_state["location"]

        dx = goal[0] - current[0]
        dy = goal[1] - current[1]
        return math.sqrt(dx*dx + dy*dy)

    def _compute_goal_direction(self) -> str:
        """Compute cardinal direction to goal."""
        if "goal_location" not in self.scenario_data:
            return "unknown"

        goal = self.scenario_data["goal_location"]
        if self.mode == "real":
            loc = self.vehicle.get_transform().location
            current = (loc.x, loc.y)
        else:
            current = (self.mock_state["location"][0], self.mock_state["location"][1])

        dx = goal[0] - current[0]
        dy = goal[1] - current[1]

        angle = math.degrees(math.atan2(dy, dx))

        if -45 <= angle < 45:
            return "east"
        elif 45 <= angle < 135:
            return "north"
        elif angle >= 135 or angle < -135:
            return "west"
        else:
            return "south"

    def close(self) -> None:
        """Cleanup resources."""
        if self.mode == "real" and self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
