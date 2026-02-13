#!/usr/bin/env python3
"""
CARLA Advanced Actions Example

Demonstrates all available actions in the CARLA environment:

Basic Actions:
- control: Manual throttle/brake/steer
- emergency_stop: Full brake
- lane_change: Change lanes (left/right)
- observe: No action (observe only)

Enhanced Actions:
- brake_vehicle: Brake with specific intensity
- maintain_speed: Maintain target speed
- lane_change with target_lane_id: Improved lane change

Navigation Actions:
- init_navigation_agent: Initialize autonomous agent
- set_destination: Set navigation destination
- follow_route: Follow planned route
"""

import asyncio
from carla_env.client import CarlaEnv
from carla_env.models import CarlaAction


async def demo_basic_actions(env):
    """Demonstrate basic actions."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Actions")
    print("="*70)

    await env.reset(scenario_name="trolley_saves")

    # Control action
    print("\n1. Manual control (throttle=0.5, steer=0.0)")
    result = await env.step(CarlaAction(
        action_type="control",
        throttle=0.5,
        steer=0.0,
        brake=0.0
    ))
    print(f"   Speed: {result.observation.speed_kmh:.1f} km/h")

    # Emergency stop
    print("\n2. Emergency stop")
    result = await env.step(CarlaAction(action_type="emergency_stop"))
    print(f"   Speed: {result.observation.speed_kmh:.1f} km/h")

    # Lane change
    print("\n3. Lane change (right)")
    result = await env.step(CarlaAction(
        action_type="lane_change",
        lane_direction="right"
    ))
    print(f"   Lane: {result.observation.current_lane}")

    # Observe
    print("\n4. Observe (no action)")
    result = await env.step(CarlaAction(action_type="observe"))
    print(f"   Speed: {result.observation.speed_kmh:.1f} km/h")


async def demo_enhanced_actions(env):
    """Demonstrate enhanced actions."""
    print("\n" + "="*70)
    print("DEMO 2: Enhanced Actions")
    print("="*70)

    await env.reset(scenario_name="trolley_saves")

    # Accelerate first
    await env.step(CarlaAction(action_type="control", throttle=0.8))
    await env.step(CarlaAction(action_type="control", throttle=0.8))

    # Brake vehicle with intensity
    print("\n1. Brake vehicle (intensity=0.5)")
    result = await env.step(CarlaAction(
        action_type="brake_vehicle",
        brake_intensity=0.5
    ))
    print(f"   Speed: {result.observation.speed_kmh:.1f} km/h")

    # Maintain speed
    print("\n2. Maintain speed (target=30 km/h)")
    for i in range(5):
        result = await env.step(CarlaAction(
            action_type="maintain_speed",
            target_speed_kmh=30.0
        ))
        print(f"   Step {i+1}: Speed={result.observation.speed_kmh:.1f} km/h")

    # Improved lane change with target_lane_id
    print("\n3. Lane change to lane_1")
    result = await env.step(CarlaAction(
        action_type="lane_change",
        target_lane_id="lane_1"
    ))
    print(f"   Lane: {result.observation.current_lane}")


async def demo_navigation_actions(env):
    """Demonstrate navigation actions."""
    print("\n" + "="*70)
    print("DEMO 3: Navigation Actions")
    print("="*70)

    result = await env.reset(scenario_name="trolley_saves")
    start_location = result.observation.location

    # Initialize navigation agent
    print("\n1. Initialize navigation agent (cautious behavior)")
    await env.step(CarlaAction(
        action_type="init_navigation_agent",
        navigation_behavior="cautious"
    ))
    print(f"   ✓ Agent initialized")

    # Set destination
    print("\n2. Set destination (100m ahead)")
    destination_x = start_location[0] + 100.0
    destination_y = start_location[1] + 50.0
    await env.step(CarlaAction(
        action_type="set_destination",
        destination_x=destination_x,
        destination_y=destination_y,
        destination_z=0.0
    ))
    print(f"   ✓ Destination: ({destination_x:.1f}, {destination_y:.1f})")

    # Follow route
    print("\n3. Follow route (5 steps)")
    for i in range(5):
        result = await env.step(CarlaAction(
            action_type="follow_route",
            route_steps=1
        ))
        obs = result.observation

        dx = destination_x - obs.location[0]
        dy = destination_y - obs.location[1]
        distance = (dx*dx + dy*dy)**0.5

        print(f"   Step {i+1}: Speed={obs.speed_kmh:.1f} km/h, "
              f"Distance={distance:.1f}m")


async def demo_mixed_actions(env):
    """Demonstrate mixing manual and autonomous actions."""
    print("\n" + "="*70)
    print("DEMO 4: Mixed Actions (Manual + Autonomous)")
    print("="*70)

    await env.reset(scenario_name="trolley_saves")

    # Manual driving
    print("\n1. Manual driving (accelerate)")
    for i in range(3):
        result = await env.step(CarlaAction(
            action_type="control",
            throttle=0.7,
            steer=0.0
        ))
        print(f"   Step {i+1}: Speed={result.observation.speed_kmh:.1f} km/h")

    # Switch to autonomous navigation
    print("\n2. Switch to autonomous navigation")
    start_location = result.observation.location
    await env.step(CarlaAction(action_type="init_navigation_agent"))
    await env.step(CarlaAction(
        action_type="set_destination",
        destination_x=start_location[0] + 50.0,
        destination_y=start_location[1] + 25.0
    ))
    print(f"   ✓ Autonomous mode activated")

    # Let agent drive
    print("\n3. Agent drives (3 steps)")
    for i in range(3):
        result = await env.step(CarlaAction(
            action_type="follow_route",
            route_steps=1
        ))
        print(f"   Step {i+1}: Speed={result.observation.speed_kmh:.1f} km/h")

    # Override with manual brake
    print("\n4. Manual override (brake)")
    result = await env.step(CarlaAction(
        action_type="brake_vehicle",
        brake_intensity=1.0
    ))
    print(f"   Speed after brake: {result.observation.speed_kmh:.1f} km/h")


async def main():
    """Run all demos."""
    print("="*70)
    print("CARLA Advanced Actions Demo")
    print("="*70)
    print("\nDemonstrating all available actions:")
    print("  - Basic actions (control, emergency_stop, lane_change, observe)")
    print("  - Enhanced actions (brake_vehicle, maintain_speed, improved lane_change)")
    print("  - Navigation actions (init_agent, set_destination, follow_route)")

    # Connect to environment
    env = CarlaEnv(base_url="http://localhost:8000")

    try:
        # Run demos
        await demo_basic_actions(env)
        await demo_enhanced_actions(env)
        await demo_navigation_actions(env)
        await demo_mixed_actions(env)

        # Show final metrics
        state = await env.state()
        print("\n" + "="*70)
        print("Final Metrics Summary")
        print("="*70)
        print(f"Total actions: {state.total_tool_calls}")
        print(f"Action breakdown:")
        for action_type, count in sorted(state.tool_call_counts.items()):
            print(f"  {action_type}: {count}")
        print(f"\nPerformance:")
        print(f"  Average speed: {state.average_speed:.1f} km/h")
        print(f"  Max speed: {state.max_speed:.1f} km/h")
        print(f"  Total distance: {state.total_distance:.2f}m")
        print(f"  Collisions: {state.collisions_count}")

    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
