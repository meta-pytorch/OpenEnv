#!/usr/bin/env python3
"""
CARLA Navigation Example - Day 4

Demonstrates how to use the navigation actions:
- init_navigation_agent: Initialize a navigation agent with behavior
- set_destination: Set a destination coordinate
- follow_route: Follow the route to destination

This example shows autonomous navigation using CARLA's BehaviorAgent.
"""

import asyncio
from carla_env.client import CarlaEnv
from carla_env.models import CarlaAction


async def main():
    """Demonstrate navigation actions."""
    # Connect to environment
    env = CarlaEnv(base_url="http://localhost:8000")

    print("="*70)
    print("CARLA Navigation Example")
    print("="*70)

    # Reset environment
    result = await env.reset(scenario_name="trolley_saves")
    obs = result.observation
    print(f"\n1. Reset environment")
    print(f"   Location: ({obs.location[0]:.1f}, {obs.location[1]:.1f}, {obs.location[2]:.1f})")
    print(f"   Speed: {obs.speed_kmh:.1f} km/h")

    # Step 1: Initialize navigation agent
    print(f"\n2. Initialize navigation agent (normal behavior)")
    result = await env.step(CarlaAction(
        action_type="init_navigation_agent",
        navigation_behavior="normal"  # Options: "cautious", "normal", "aggressive"
    ))
    print(f"   ✓ Agent initialized")

    # Step 2: Set destination
    print(f"\n3. Set destination (50m ahead)")
    start_x = obs.location[0]
    start_y = obs.location[1]
    destination_x = start_x + 50.0
    destination_y = start_y + 25.0

    result = await env.step(CarlaAction(
        action_type="set_destination",
        destination_x=destination_x,
        destination_y=destination_y,
        destination_z=0.0
    ))
    print(f"   ✓ Destination set to ({destination_x:.1f}, {destination_y:.1f})")

    # Step 3: Follow route
    print(f"\n4. Follow route (autonomous navigation)")
    for step in range(10):
        result = await env.step(CarlaAction(
            action_type="follow_route",
            route_steps=1
        ))
        obs = result.observation

        # Compute distance to destination
        dx = destination_x - obs.location[0]
        dy = destination_y - obs.location[1]
        distance = (dx*dx + dy*dy)**0.5

        print(f"   Step {step + 1}: "
              f"Location=({obs.location[0]:.1f}, {obs.location[1]:.1f}), "
              f"Speed={obs.speed_kmh:.1f} km/h, "
              f"Distance to goal={distance:.1f}m")

        if distance < 5.0:
            print(f"   ✓ Reached destination!")
            break

        if result.observation.done:
            print(f"   Episode terminated: {result.observation.done_reason}")
            break

    # Get final state with metrics
    state = await env.state()
    print(f"\n5. Final metrics:")
    print(f"   Total turns: {state.num_turns}")
    print(f"   Total distance: {state.total_distance:.2f}m")
    print(f"   Average speed: {state.average_speed:.1f} km/h")
    print(f"   Max speed: {state.max_speed:.1f} km/h")
    print(f"   Action counts: {state.tool_call_counts}")

    # Cleanup
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
