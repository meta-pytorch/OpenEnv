#!/usr/bin/env python3
"""
CARLA Maze Navigation Example

Demonstrates the maze_navigation scenario with goal-directed navigation.

The maze scenario tests basic navigation abilities:
- Vehicle spawns at origin
- Goal is set ~150m away (diagonal, not straight ahead)
- Vehicle must navigate to goal using position and goal information
- Episode ends when goal is reached (within 5m) or timeout (200 steps)

This is the simplest navigation test - no obstacles, just driving to a destination.
Based on sinatras/carla-env maze scenario concept.
"""

import asyncio
import math
from carla_env.client import CarlaEnv
from carla_env.models import CarlaAction


async def demo_manual_navigation():
    """Demonstrate manual navigation to goal using control actions."""
    print("=" * 70)
    print("DEMO 1: Manual Navigation (control actions)")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    # Reset with maze scenario
    result = await env.reset(scenario_name="maze_navigation")
    obs = result.observation

    print(f"\n📍 Initial State:")
    print(f"   Location: ({obs.location[0]:.1f}, {obs.location[1]:.1f})")
    print(f"   Speed: {obs.speed_kmh:.1f} km/h")
    print(f"   Goal distance: {obs.goal_distance:.1f}m")
    print(f"   Goal direction: {obs.goal_direction}")
    print(f"   Scene: {obs.scene_description}")

    # Simple navigation: drive toward goal
    print(f"\n🚗 Navigating to goal...")
    for step in range(50):
        # Compute steering toward goal
        goal_x = obs.location[0] + obs.goal_distance * 0.7  # Approximate
        goal_y = obs.location[1] + obs.goal_distance * 0.7
        current_x, current_y = obs.location[0], obs.location[1]

        # Simple proportional steering
        dx = goal_x - current_x
        dy = goal_y - current_y
        target_angle = math.degrees(math.atan2(dy, dx))
        current_yaw = obs.rotation[1]

        angle_diff = target_angle - current_yaw
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # Steering: proportional to angle difference
        steer = max(-1.0, min(1.0, angle_diff / 45.0))

        # Throttle: slow down near goal
        if obs.goal_distance < 10.0:
            throttle = 0.3
        else:
            throttle = 0.5

        # Execute control
        result = await env.step(CarlaAction(
            action_type="control",
            throttle=throttle,
            steer=steer
        ))
        obs = result.observation

        if step % 5 == 0:
            print(f"   Step {step:3d}: "
                  f"Location=({obs.location[0]:6.1f}, {obs.location[1]:6.1f}), "
                  f"Goal={obs.goal_distance:5.1f}m {obs.goal_direction:5s}, "
                  f"Speed={obs.speed_kmh:5.1f} km/h")

        if obs.done:
            print(f"\n   ✓ Episode ended: {obs.done_reason}")
            break

    # Show final metrics
    state = await env.state()
    print(f"\n📊 Final Metrics:")
    print(f"   Total distance traveled: {state.total_distance:.1f}m")
    print(f"   Total reward: {state.total_reward:.2f}")
    print(f"   Average speed: {state.average_speed:.1f} km/h")
    print(f"   Total steps: {state.num_turns}")

    await env.close()


async def demo_autonomous_navigation():
    """Demonstrate autonomous navigation using navigation agent."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Autonomous Navigation (navigation agent)")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    # Reset with maze scenario
    result = await env.reset(scenario_name="maze_navigation")
    obs = result.observation

    print(f"\n📍 Initial State:")
    print(f"   Location: ({obs.location[0]:.1f}, {obs.location[1]:.1f})")
    print(f"   Goal distance: {obs.goal_distance:.1f}m")
    print(f"   Goal direction: {obs.goal_direction}")

    # Initialize navigation agent
    print(f"\n🤖 Initializing navigation agent...")
    await env.step(CarlaAction(
        action_type="init_navigation_agent",
        navigation_behavior="normal"
    ))
    print(f"   ✓ Agent initialized")

    # Set destination to goal
    # In maze scenario, goal is at ~(105, 105, 0.5) for default config
    # We can compute from goal_distance and direction
    print(f"\n🎯 Setting destination to goal location...")
    goal_location = (105.0, 105.0, 0.5)  # Default maze goal
    await env.step(CarlaAction(
        action_type="set_destination",
        destination_x=goal_location[0],
        destination_y=goal_location[1],
        destination_z=goal_location[2]
    ))
    print(f"   ✓ Destination set to {goal_location}")

    # Follow route autonomously
    print(f"\n🚗 Following route autonomously...")
    for step in range(50):
        result = await env.step(CarlaAction(
            action_type="follow_route",
            route_steps=1
        ))
        obs = result.observation

        if step % 5 == 0:
            print(f"   Step {step:3d}: "
                  f"Goal={obs.goal_distance:5.1f}m {obs.goal_direction:5s}, "
                  f"Speed={obs.speed_kmh:5.1f} km/h")

        if obs.done:
            print(f"\n   ✓ Episode ended: {obs.done_reason}")
            break

    # Show final metrics
    state = await env.state()
    print(f"\n📊 Final Metrics:")
    print(f"   Total distance traveled: {state.total_distance:.1f}m")
    print(f"   Total reward: {state.total_reward:.2f}")
    print(f"   Average speed: {state.average_speed:.1f} km/h")
    print(f"   Total steps: {state.num_turns}")

    await env.close()


async def main():
    """Run maze navigation demos."""
    print("\n" + "=" * 70)
    print("CARLA Maze Navigation Example")
    print("=" * 70)
    print("\nThe maze scenario is the simplest navigation test:")
    print("  - Vehicle starts at origin (0, 0)")
    print("  - Goal is ~150m away diagonally")
    print("  - No obstacles or other actors")
    print("  - Success = reach goal within 5m")
    print("  - Timeout = 200 steps")
    print("\nDemonstrates two approaches:")
    print("  1. Manual navigation using control actions")
    print("  2. Autonomous navigation using navigation agent")

    try:
        await demo_manual_navigation()
        await demo_autonomous_navigation()

        print("\n\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print("\nThe maze scenario tests basic goal-directed navigation:")
        print("  ✓ Goal distance/direction tracking")
        print("  ✓ Manual control for simple navigation")
        print("  ✓ Autonomous navigation agent integration")
        print("\nThis provides the foundation for more complex navigation tasks.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Make sure the CARLA environment server is running:")
        print("  docker run -p 8000:8000 carla-env:latest")


if __name__ == "__main__":
    asyncio.run(main())
