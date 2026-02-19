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
from typing import Optional, Dict, Any, List
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


class CollisionSensor:
    """Collision sensor that tracks unique collisions."""

    def __init__(self, world, vehicle):
        self._world = world
        self._vehicle = vehicle
        self._sensor = None
        self._collided_actors = {}

    def setup(self):
        """Create and configure the collision sensor."""
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self._sensor = self._world.try_spawn_actor(blueprint, transform, attach_to=self._vehicle)

        if self._sensor is None:
            raise RuntimeError("Failed to spawn collision sensor")

        self._sensor.listen(self._on_collision)

    def _on_collision(self, event):
        """Record collision with unique actor."""
        try:
            if event.other_actor:
                actor_id = int(event.other_actor.id)
                actor_type = str(event.other_actor.type_id)
                self._collided_actors[actor_id] = actor_type
        except Exception:
            pass  # Silently ignore collision parsing errors

    def count_unique_by_prefix(self, prefix: str) -> int:
        """Count unique actors hit that match prefix (e.g., 'walker.')."""
        return sum(1 for type_id in self._collided_actors.values() if type_id.startswith(prefix))

    @property
    def collision_count(self) -> int:
        """Total number of unique collisions detected."""
        return len(self._collided_actors)

    @property
    def events(self):
        """Get collision events."""
        # Convert our dict format to event-like format
        return [
            {"actor_id": actor_id, "actor_type": actor_type}
            for actor_id, actor_type in self._collided_actors.items()
        ]

    def reset(self):
        """Clear collision history."""
        self._collided_actors.clear()

    def destroy(self):
        """Clean up sensor."""
        if self._sensor:
            try:
                if self._sensor.is_alive:
                    self._sensor.stop()
                self._sensor.destroy()
            except:
                pass
            self._sensor = None


class WorldWrapper:
    """Wrapper to provide runtime.world.world access pattern."""

    def __init__(self, world):
        self.world = world  # CARLA World object

    def get_map(self):
        return self.world.get_map()


class ActorsHelper:
    """Helper for spawning actors in scenarios."""

    def __init__(self, world):
        self.world = world
        self._spawned_actors = []

    def spawn_pedestrian(self, transform):
        """Spawn a pedestrian at the given transform."""
        try:
            blueprint_library = self.world.get_blueprint_library()
            pedestrian_bps = blueprint_library.filter('walker.pedestrian.*')
            if not pedestrian_bps:
                return None

            pedestrian_bp = pedestrian_bps[0]
            # Make pedestrian vulnerable to collisions
            if pedestrian_bp.has_attribute("is_invincible"):
                pedestrian_bp.set_attribute("is_invincible", "false")

            actor = self.world.try_spawn_actor(pedestrian_bp, transform)

            if actor is not None:
                self._spawned_actors.append(actor)

            return actor
        except Exception as e:
            return None

    def cleanup(self):
        """Destroy all spawned actors."""
        for actor in self._spawned_actors:
            if actor is not None:
                try:
                    actor.destroy()
                except:
                    pass
        self._spawned_actors.clear()


class CarlaRuntime:
    """Runtime object that scenarios expect."""

    def __init__(self, world, vehicle, client, collision_sensor, actors_helper):
        self.world = WorldWrapper(world)  # Wrapped to support runtime.world.world
        self.world_obj = world  # Direct reference
        self.ego_vehicle = vehicle
        self.client = client
        self.map = world.get_map()
        self.collision_sensor = collision_sensor
        self.actors = actors_helper  # For spawning pedestrians

    def get_map(self):
        """Get CARLA map."""
        return self.map


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

        # Navigation agent (real mode only)
        self.nav_agent: Optional[Any] = None

        # Mock mode state
        self.mock_state: Dict[str, Any] = {}

        # Scenario data
        self.scenario_data: Dict[str, Any] = {}

    def reset(self, scenario_name: Optional[str] = None) -> CarlaObservation:
        """
        Reset environment and setup scenario.

        Args:
            scenario_name: Optional scenario name to switch to. If None, uses current scenario.

        Returns:
            Initial observation
        """
        # Switch scenario if requested
        if scenario_name is not None and scenario_name != self.scenario.name:
            self.scenario = get_scenario(scenario_name)

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

        In real mode: Apply control to CARLA vehicle and tick world
        In mock mode: Update simulated physics

        Args:
            action: Action to execute

        Returns:
            Observation after action
        """
        # Safety net for the HTTP REST path (POST /step), which creates a
        # fresh CarlaEnvironment per request and may call step() before reset().
        # The WebSocket path keeps one env per session so this rarely triggers.
        if self.mode == "real" and (self.world is None or self.vehicle is None):
            self.reset()

        # capture_image is a read-only operation: return the latest buffered
        # camera frame without advancing the simulation or counting as a step.
        if action.action_type == "capture_image":
            obs = self._get_observation()
            if self.mode == "real":
                camera_image = self.capture_image()
                if camera_image:
                    obs.camera_image = camera_image
            return obs

        # Increment step counter
        self._state.step_count += 1

        # Track action metrics
        self._state.num_turns += 1
        self._state.total_tool_calls += 1

        # Track action type count
        action_name = action.action_type
        if action_name not in self._state.tool_call_counts:
            self._state.tool_call_counts[action_name] = 0
        self._state.tool_call_counts[action_name] += 1

        # Store previous state for distance tracking
        if self.mode == "real" and self.vehicle is not None:
            prev_location = self.vehicle.get_location()
            prev_speed = self._get_current_speed()
        else:
            prev_location = None
            prev_speed = self.mock_state.get("speed_kmh", 0.0) if hasattr(self, "mock_state") else 0.0

        # Execute action
        if self.mode == "real":
            self._step_real_mode(action)
        else:
            self._step_mock_mode(action)

        # Track distance and speed after action
        if self.mode == "real" and self.vehicle is not None:
            new_location = self.vehicle.get_location()
            if prev_location is not None:
                distance = prev_location.distance(new_location)
                self._state.total_distance += distance

            # Track speed
            current_speed = self._get_current_speed()
            self._state.max_speed = max(self._state.max_speed, current_speed)

            # Update average speed (running average)
            if self._state.num_turns > 0:
                self._state.average_speed = (
                    (self._state.average_speed * (self._state.num_turns - 1) + current_speed)
                    / self._state.num_turns
                )
        else:
            # Mock mode tracking
            current_speed = self.mock_state.get("speed_kmh", 0.0) if hasattr(self, "mock_state") else 0.0
            self._state.max_speed = max(self._state.max_speed, current_speed)

            if self._state.num_turns > 0:
                self._state.average_speed = (
                    (self._state.average_speed * (self._state.num_turns - 1) + current_speed)
                    / self._state.num_turns
                )

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

        # Assign reward to observation before returning
        obs.reward = reward

        return obs

    @property
    def state(self) -> CarlaState:
        """Get current episode state."""
        return self._state

    def _find_best_spawn_point(
        self,
        spawn_points: List[Any],
        carla_map: Any,
        min_forward_m: float = 35.0,
        require_left: bool = False,
        require_right: bool = False,
        max_angle_deg: float = 15.0,
    ) -> Any:
        """
        Find a spawn point with a straight road ahead and required lane topology.

        Scores each spawn point by checking that the road 'min_forward_m' meters
        ahead stays within 'max_angle_deg' of the vehicle's forward direction.
        Also checks adjacent lane availability when required by the scenario.

        Args:
            spawn_points: CARLA spawn point transforms
            carla_map: CARLA map for waypoint queries
            min_forward_m: How far ahead the road must be straight
            require_left: Scenario needs a left adjacent lane
            require_right: Scenario needs a right adjacent lane
            max_angle_deg: Maximum deviation angle to consider "straight"

        Returns:
            Best spawn point transform
        """
        from .benchmark_scenarios.shared import same_direction

        best_transform = None
        best_score = float("inf")  # lower is better (angle deviation)

        for sp in spawn_points:
            wp = carla_map.get_waypoint(
                sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp is None:
                continue

            # Check adjacent lane requirements
            if require_left:
                left = wp.get_left_lane()
                if left is None or left.lane_type != carla.LaneType.Driving:
                    continue
                if not same_direction(wp, left):
                    continue

            if require_right:
                right = wp.get_right_lane()
                if right is None or right.lane_type != carla.LaneType.Driving:
                    continue
                if not same_direction(wp, right):
                    continue

            # Check road straightness: get waypoint min_forward_m ahead
            ahead_list = wp.next(min_forward_m)
            if not ahead_list:
                continue
            ahead_wp = ahead_list[0]

            # Compute angle between spawn forward vector and direction to ahead waypoint
            fwd = sp.get_forward_vector()
            dx = ahead_wp.transform.location.x - sp.location.x
            dy = ahead_wp.transform.location.y - sp.location.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.0:
                continue  # degenerate

            # Dot product gives cosine of angle
            cos_angle = (fwd.x * dx + fwd.y * dy) / dist
            cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp
            angle_deg = math.degrees(math.acos(cos_angle))

            if angle_deg > max_angle_deg:
                continue  # road curves too much

            # Also check a midpoint to catch S-curves
            mid_list = wp.next(min_forward_m / 2.0)
            if mid_list:
                mid_wp = mid_list[0]
                mdx = mid_wp.transform.location.x - sp.location.x
                mdy = mid_wp.transform.location.y - sp.location.y
                mdist = math.sqrt(mdx * mdx + mdy * mdy)
                if mdist > 1.0:
                    mid_cos = (fwd.x * mdx + fwd.y * mdy) / mdist
                    mid_cos = max(-1.0, min(1.0, mid_cos))
                    mid_angle = math.degrees(math.acos(mid_cos))
                    if mid_angle > max_angle_deg:
                        continue

            # Score: prefer smallest angle (straightest road)
            if angle_deg < best_score:
                best_score = angle_deg
                best_transform = sp

        return best_transform

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
        if hasattr(self, '_spawned_simple_actors'):
            for actor in self._spawned_simple_actors:
                if actor is not None:
                    try:
                        actor.destroy()
                    except:
                        pass
            self._spawned_simple_actors = []

        if hasattr(self, 'actors_helper') and self.actors_helper is not None:
            self.actors_helper.cleanup()
            self.actors_helper = None

        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None

        # Reset navigation agent
        self.nav_agent = None

        # Set weather
        weather_name = setup.get("weather", "ClearNoon")
        weather = getattr(carla.WeatherParameters, weather_name)
        self.world.set_weather(weather)

        # Spawn vehicle
        spawn_point = setup.get("spawn_point", {})
        loc = spawn_point.get("location", None)
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

        # Use CARLA's spawn points if location not specified (avoids sidewalks)
        if loc is None:
            carla_map = self.world.get_map()
            spawn_points = carla_map.get_spawn_points()
            if spawn_points:
                # Determine lane requirements from scenario
                require_left = False
                require_right = False
                min_forward_m = 35.0

                from .scenario_adapter import SinatrasScenarioAdapter
                from .scenarios import SimpleTrolleyScenario
                if isinstance(self.scenario, SinatrasScenarioAdapter):
                    sinatras = self.scenario.sinatras_scenario
                    if hasattr(sinatras, "spawn_requirements"):
                        reqs = sinatras.spawn_requirements()
                        require_left = reqs.get("require_left", False)
                        require_right = reqs.get("require_right", False)
                        min_forward_m = max(35.0, reqs.get("min_forward_m", 35.0))
                elif isinstance(self.scenario, SimpleTrolleyScenario):
                    if self.scenario.pedestrians_adjacent > 0:
                        require_left = True

                # Find best spawn point: straight road + required lanes
                transform = self._find_best_spawn_point(
                    spawn_points, carla_map,
                    min_forward_m=min_forward_m,
                    require_left=require_left,
                    require_right=require_right,
                )

                if transform is None:
                    # Relax: try without lane requirements
                    transform = self._find_best_spawn_point(
                        spawn_points, carla_map,
                        min_forward_m=min_forward_m,
                    )

                if transform is None:
                    # Final fallback: first spawn point
                    transform = spawn_points[0]
            else:
                # Fallback to origin if no spawn points available
                transform = carla.Transform(
                    carla.Location(x=0.0, y=0.0, z=0.5),
                    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
                )
        else:
            # Use provided location
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

        # Create collision sensor BEFORE spawning any actors (needs to be listening from the start)
        self.collision_sensor = CollisionSensor(self.world, self.vehicle)
        self.collision_sensor.setup()  # Two-phase init

        # Create camera sensor for image capture
        self.camera_sensor = None
        self.latest_camera_image = None
        try:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            # Lower resolution to fit WebSocket message limit (1MB)
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '360')
            camera_bp.set_attribute('fov', '90')
            # Mount camera on hood looking forward
            camera_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
            self.camera_sensor = self.world.try_spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            if self.camera_sensor:
                self.camera_sensor.listen(lambda image: self._on_camera_image(image))
        except Exception:
            pass

        # Spawn actors from setup
        actors_to_spawn = setup.get("actors", [])
        if actors_to_spawn:
            blueprint_library = self.world.get_blueprint_library()
            pedestrian_bps = blueprint_library.filter('walker.pedestrian.*')

            spawned_count = 0
            for actor_def in actors_to_spawn:
                if actor_def.get("type") == "pedestrian":
                    try:
                        # Get pedestrian blueprint
                        if pedestrian_bps:
                            ped_bp = pedestrian_bps[0]
                            # Make pedestrian vulnerable to collisions
                            if ped_bp.has_attribute("is_invincible"):
                                ped_bp.set_attribute("is_invincible", "false")
                        else:
                            continue

                        # Calculate spawn location relative to vehicle
                        vehicle_transform = self.vehicle.get_transform()
                        distance = actor_def.get("distance", 25.0)
                        lane_offset = actor_def.get("lane_offset", 0.0)

                        # Calculate location
                        forward = vehicle_transform.get_forward_vector()
                        right = vehicle_transform.get_right_vector()

                        spawn_loc = carla.Location(
                            x=vehicle_transform.location.x + forward.x * distance + right.x * lane_offset,
                            y=vehicle_transform.location.y + forward.y * distance + right.y * lane_offset,
                            z=vehicle_transform.location.z + 0.5
                        )

                        # Spawn pedestrian
                        spawn_transform = carla.Transform(spawn_loc, vehicle_transform.rotation)
                        pedestrian = self.world.try_spawn_actor(ped_bp, spawn_transform)

                        if pedestrian is not None:
                            # Track spawned actors for cleanup
                            if not hasattr(self, '_spawned_simple_actors'):
                                self._spawned_simple_actors = []
                            self._spawned_simple_actors.append(pedestrian)
                            spawned_count += 1

                    except Exception:
                        # Spawn can fail if location is occupied, continue with others
                        pass

            # Tick to ensure actors are spawned
            self.world.tick()

        # If scenario is a sinatras adapter, create runtime and call setup_carla()
        from .scenario_adapter import SinatrasScenarioAdapter
        if isinstance(self.scenario, SinatrasScenarioAdapter):
            # Create actors helper for spawning pedestrians
            self.actors_helper = ActorsHelper(self.world)

            # Create runtime with all helpers
            runtime = CarlaRuntime(
                self.world,
                self.vehicle,
                self.client,
                self.collision_sensor,
                self.actors_helper
            )
            self.scenario.setup_carla(runtime)

    def _reset_mock_mode(self, setup: Dict[str, Any]) -> None:
        """Reset in mock simulation mode."""
        # Initialize mock state
        spawn_point = setup.get("spawn_point", {})
        loc = spawn_point.get("location") or (0.0, 0.0, 0.5)
        rot = spawn_point.get("rotation") or (0.0, 0.0, 0.0)

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

        # Reset navigation agent (mock)
        self.nav_agent = None

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

        elif action.action_type == "brake_vehicle":
            # Brake with specific intensity
            # Adapted from sinatras/carla-env tools/vehicle.py:brake_vehicle()
            intensity = action.brake_intensity if action.brake_intensity is not None else 1.0
            intensity = max(0.0, min(1.0, float(intensity)))  # Clamp [0.0, 1.0]
            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=intensity,
                hand_brake=False
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "maintain_speed":
            # Maintain target speed with simple PID-like control
            target_speed = action.target_speed_kmh if action.target_speed_kmh is not None else 30.0
            current_speed = self._get_current_speed()

            # Simple proportional control
            speed_error = target_speed - current_speed
            if speed_error > 2.0:  # Need to accelerate
                throttle = min(0.5, speed_error * 0.05)
                brake_val = 0.0
            elif speed_error < -2.0:  # Need to brake
                throttle = 0.0
                brake_val = min(0.5, abs(speed_error) * 0.05)
            else:  # Close enough, coast
                throttle = 0.1
                brake_val = 0.0

            control = carla.VehicleControl(
                throttle=throttle,
                steer=0.0,
                brake=brake_val
            )
            self.vehicle.apply_control(control)

        elif action.action_type == "lane_change":
            # Improved lane change with target_lane_id support
            # Backward compatible with lane_direction
            if action.target_lane_id:
                # New way: use target_lane_id (e.g., "lane_1", "lane_0")
                # For now, simple implementation: steer based on lane number
                current_lane = self.current_lane if hasattr(self, 'current_lane') else "lane_0"
                target_lane = action.target_lane_id

                # Extract lane numbers (assuming format "lane_N")
                try:
                    current_num = int(current_lane.split('_')[1]) if '_' in current_lane else 0
                    target_num = int(target_lane.split('_')[1]) if '_' in target_lane else 0
                    lane_diff = target_num - current_num

                    # Steer proportional to lane difference
                    steer = -0.3 if lane_diff < 0 else 0.3 if lane_diff > 0 else 0.0
                except (IndexError, ValueError):
                    steer = 0.0
            else:
                # Old way: use lane_direction for backward compatibility
                steer = -0.5 if action.lane_direction == "left" else 0.5

            control = carla.VehicleControl(throttle=0.3, steer=steer)
            self.vehicle.apply_control(control)

        elif action.action_type == "observe":
            # No-op: just observe without changing control
            # This is the default action type for backward compatibility
            pass

        elif action.action_type == "init_navigation_agent":
            # Initialize navigation agent
            behavior = action.navigation_behavior if action.navigation_behavior else "normal"

            # Import agents (lazy import - only when needed)
            from carla_env.server._carla_agents.navigation.behavior_agent import BehaviorAgent
            from carla_env.server._carla_agents.navigation.basic_agent import BasicAgent

            # Create agent based on behavior
            if behavior == "normal":
                self.nav_agent = BehaviorAgent(self.vehicle, behavior=behavior)
            elif behavior in ["cautious", "aggressive"]:
                self.nav_agent = BehaviorAgent(self.vehicle, behavior=behavior)
            else:
                # Fallback to BasicAgent for unknown behaviors
                self.nav_agent = BasicAgent(self.vehicle)

        elif action.action_type == "set_destination":
            # Set destination for navigation agent
            if self.nav_agent is None:
                # Auto-initialize with normal behavior if not initialized
                from carla_env.server._carla_agents.navigation.behavior_agent import BehaviorAgent
                self.nav_agent = BehaviorAgent(self.vehicle, behavior="normal")

            # Set destination
            if action.destination_x is not None and action.destination_y is not None:
                z = action.destination_z if action.destination_z is not None else 0.0
                destination = carla.Location(
                    x=action.destination_x,
                    y=action.destination_y,
                    z=z
                )
                self.nav_agent.set_destination(destination)

        elif action.action_type == "follow_route":
            # Follow route using navigation agent
            if self.nav_agent is None:
                # No agent initialized - just maintain current control
                pass
            else:
                # Execute navigation for specified steps
                steps = action.route_steps if action.route_steps else 1
                for _ in range(steps):
                    if not self.nav_agent.done():
                        control = self.nav_agent.run_step()
                        self.vehicle.apply_control(control)
                        self.world.tick()
                    else:
                        # Reached destination
                        break

        # Tick simulation (unless already ticked by follow_route)
        if action.action_type != "follow_route":
            self.world.tick()

        # Update collision state after tick
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            if hasattr(self.collision_sensor, '_collided_actors'):
                # Add new collisions to state.collisions
                for actor_id, actor_type in self.collision_sensor._collided_actors.items():
                    # Check if this collision is already recorded
                    existing = any(c.get("actor_id") == actor_id for c in self._state.collisions)
                    if not existing:
                        collision = {
                            "frame": self._state.step_count,
                            "actor_id": actor_id,
                            "actor_type": actor_type,
                            "intensity": self._get_current_speed(),
                        }
                        self._state.collisions.append(collision)
                        self._state.collisions_count += 1
                        self._state.collision_intensity_total += self._get_current_speed()

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

        elif action.action_type == "brake_vehicle":
            # Brake with specific intensity
            intensity = action.brake_intensity if action.brake_intensity is not None else 1.0
            intensity = max(0.0, min(1.0, float(intensity)))
            # Apply deceleration proportional to intensity
            decel = intensity * 8.0  # m/s^2
            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms - decel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "maintain_speed":
            # Maintain target speed
            target_speed = action.target_speed_kmh if action.target_speed_kmh is not None else 30.0
            current_speed = self.mock_state["speed_kmh"]
            speed_error = target_speed - current_speed

            # Simple proportional control
            if speed_error > 2.0:
                accel = min(3.0, speed_error * 0.5)
            elif speed_error < -2.0:
                accel = max(-8.0, speed_error * 0.5)
            else:
                accel = 0.0

            speed_ms = self.mock_state["speed_kmh"] / 3.6
            speed_ms = max(0.0, speed_ms + accel * dt)
            self.mock_state["speed_kmh"] = speed_ms * 3.6

        elif action.action_type == "lane_change":
            # Improved with target_lane_id support
            # Lateral offset (simplified)
            if action.target_lane_id:
                # New way: use target_lane_id
                offset = -3.5 if "0" in action.target_lane_id else 3.5
            else:
                # Old way: backward compatible
                offset = -3.5 if action.lane_direction == "left" else 3.5

            yaw_rad = math.radians(self.mock_state["rotation"][1])
            self.mock_state["location"][0] += offset * math.sin(yaw_rad)
            self.mock_state["location"][1] += offset * math.cos(yaw_rad)

        elif action.action_type == "observe":
            # No-op: just observe without changing state
            # This is the default action type for backward compatibility
            pass

        elif action.action_type == "init_navigation_agent":
            # Mock navigation agent initialization
            # Store navigation config in mock state
            behavior = action.navigation_behavior if action.navigation_behavior else "normal"
            self.mock_state["nav_agent"] = {
                "initialized": True,
                "behavior": behavior,
                "destination": None,
            }

        elif action.action_type == "set_destination":
            # Mock set destination
            if "nav_agent" not in self.mock_state:
                self.mock_state["nav_agent"] = {
                    "initialized": True,
                    "behavior": "normal",
                    "destination": None,
                }

            if action.destination_x is not None and action.destination_y is not None:
                z = action.destination_z if action.destination_z is not None else 0.0
                self.mock_state["nav_agent"]["destination"] = (
                    action.destination_x,
                    action.destination_y,
                    z
                )

        elif action.action_type == "follow_route":
            # Mock follow route
            # Simple simulation: move towards destination
            if "nav_agent" in self.mock_state and self.mock_state["nav_agent"]["destination"]:
                dest = self.mock_state["nav_agent"]["destination"]
                current = self.mock_state["location"]

                # Compute direction to destination
                dx = dest[0] - current[0]
                dy = dest[1] - current[1]
                distance = math.sqrt(dx*dx + dy*dy)

                if distance > 1.0:
                    # Move towards destination
                    speed = 30.0  # km/h
                    speed_ms = speed / 3.6

                    # Normalize direction
                    dx /= distance
                    dy /= distance

                    # Move
                    steps = action.route_steps if action.route_steps else 1
                    for _ in range(steps):
                        self.mock_state["location"][0] += dx * speed_ms * dt
                        self.mock_state["location"][1] += dy * speed_ms * dt
                        self.mock_state["time"] += dt

                    self.mock_state["speed_kmh"] = speed

                    # Update rotation to face destination
                    angle = math.degrees(math.atan2(dy, dx))
                    self.mock_state["rotation"][1] = angle

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

                        # Track collision metrics
                        self._state.collisions_count += 1
                        self._state.collision_intensity_total += self.mock_state["speed_kmh"]

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

    def _get_current_speed(self) -> float:
        """Get current speed in km/h."""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6  # Convert m/s to km/h

    def _get_observation_real(self) -> CarlaObservation:
        """Get observation from real CARLA."""
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Check collision sensor if it exists
        collision_detected = False
        collided_with = None
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            # Check if any collisions occurred (_collided_actors is now a dict: actor_id -> type_id)
            if hasattr(self.collision_sensor, '_collided_actors'):
                collision_detected = len(self.collision_sensor._collided_actors) > 0
                if collision_detected:
                    # Return first collided actor type (from dict values)
                    collided_with = list(self.collision_sensor._collided_actors.values())[0]

        # Compute goal info if goal is set
        goal_dist = self._compute_goal_distance()
        goal_dir = self._compute_goal_direction()

        return CarlaObservation(
            speed_kmh=speed_kmh,
            location=(transform.location.x, transform.location.y, transform.location.z),
            rotation=(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            current_lane="lane_0",  # Simplified
            nearby_actors=self._get_nearby_actors_real(),
            collision_detected=collision_detected,
            collided_with=collided_with,
            goal_distance=goal_dist if goal_dist != float("inf") else None,
            goal_direction=goal_dir if goal_dir != "unknown" else None,
        )

    def _get_observation_mock(self) -> CarlaObservation:
        """Get observation from mock state."""
        collision_detected = len(self.mock_state["collisions"]) > 0
        collided_with = None
        if collision_detected:
            collided_with = self.mock_state["collisions"][-1]["actor_id"]

        # Compute goal info if goal is set
        goal_dist = self._compute_goal_distance()
        goal_dir = self._compute_goal_direction()

        return CarlaObservation(
            speed_kmh=self.mock_state["speed_kmh"],
            location=tuple(self.mock_state["location"]),
            rotation=tuple(self.mock_state["rotation"]),
            current_lane="lane_0",
            nearby_actors=self._get_nearby_actors_mock(),
            collision_detected=collision_detected,
            collided_with=collided_with,
            goal_distance=goal_dist if goal_dist != float("inf") else None,
            goal_direction=goal_dir if goal_dir != "unknown" else None,
        )

    def _get_nearby_actors_real(self) -> list:
        """Get nearby actors from CARLA world."""
        try:
            world_actors = self.world.get_actors()
            ego_location = self.vehicle.get_transform().location
            ego_forward = self.vehicle.get_transform().get_forward_vector()

            nearby = []
            for actor in world_actors:
                # Skip self
                if actor.id == self.vehicle.id:
                    continue

                # Only include pedestrians and vehicles
                actor_type = actor.type_id
                if not (actor_type.startswith('walker.') or actor_type.startswith('vehicle.')):
                    continue

                # Calculate distance and position relative to ego
                actor_location = actor.get_transform().location
                distance = actor_location.distance(ego_location)

                # Only include actors within 50m
                if distance > 50.0:
                    continue

                # Determine position (ahead, behind, left, right)
                dx = actor_location.x - ego_location.x
                dy = actor_location.y - ego_location.y

                # Project onto forward vector to determine ahead/behind
                forward_dist = dx * ego_forward.x + dy * ego_forward.y

                if forward_dist > 0:
                    position = "ahead"
                else:
                    position = "behind"

                nearby.append({
                    "type": actor_type,
                    "id": actor.id,
                    "distance": distance,
                    "position": position,
                })

            return nearby

        except Exception:
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

    def _on_camera_image(self, image):
        """Callback for camera sensor - stores latest image."""
        import numpy as np
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))  # BGRA
        array = array[:, :, :3]  # Drop alpha, keep BGR
        array = array[:, :, ::-1]  # BGR to RGB
        self.latest_camera_image = array

    def capture_image(self):
        """Return the latest buffered camera image as base64.

        Does not tick the world or advance the simulation — the camera
        sensor callback continuously updates ``latest_camera_image`` on
        every world tick, so this just encodes whatever was last captured.
        """
        if self.mode != "real" or self.camera_sensor is None:
            return None

        if self.latest_camera_image is None:
            return None

        import io
        import base64
        from PIL import Image

        img = Image.fromarray(self.latest_camera_image)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=75)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def close(self) -> None:
        """Cleanup resources."""
        if self.mode == "real":
            # Cleanup simple scenario actors
            if hasattr(self, '_spawned_simple_actors'):
                for actor in self._spawned_simple_actors:
                    if actor is not None:
                        try:
                            actor.destroy()
                        except:
                            pass
                self._spawned_simple_actors = []

            # Cleanup spawned actors
            if hasattr(self, 'actors_helper') and self.actors_helper is not None:
                self.actors_helper.cleanup()
                self.actors_helper = None

            # Cleanup collision sensor if exists
            if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
                self.collision_sensor.destroy()
                self.collision_sensor = None

            # Cleanup camera sensor if exists
            if hasattr(self, 'camera_sensor') and self.camera_sensor is not None:
                try:
                    if self.camera_sensor.is_alive:
                        self.camera_sensor.stop()
                    self.camera_sensor.destroy()
                except:
                    pass
                self.camera_sensor = None

            # Cleanup vehicle
            if self.vehicle is not None:
                self.vehicle.destroy()
                self.vehicle = None
