#!/usr/bin/env python3
"""
Example: CARLA Environment - Trolley Problem

This example demonstrates the CARLA environment running a trolley problem scenario.
The vehicle approaches pedestrians and must decide whether to brake, swerve, or do nothing.

This showcases embodied evaluation where:
- Time flows continuously (the car keeps moving)
- Actions have irreversible consequences
- Inaction is itself a measurable choice

Prerequisites:
    1. Start the CARLA environment server:
       ```bash
       cd envs/carla_env
       CARLA_MODE=mock CARLA_SCENARIO=trolley_saves python -m server.app
       ```

    2. Run this example:
       ```bash
       python examples/carla_env_example.py
       ```
"""

import sys
import time

# Add envs to path for local development
sys.path.insert(0, "envs")

from carla_env import CarlaEnv, CarlaAction


def run_trolley_scenario():
    """Run a trolley problem scenario with simple decision logic."""

    print("=" * 60)
    print("CARLA Environment - Trolley Problem Example")
    print("=" * 60)
    print()

    # Connect to CARLA environment server
    print("Connecting to CARLA environment server...")
    env = CarlaEnv(base_url="http://localhost:8000")

    try:
        # Reset environment
        print("\nResetting environment (scenario: trolley_saves)...")
        result = env.reset()

        print("\n" + "=" * 60)
        print("INITIAL SCENE")
        print("=" * 60)
        print(result.observation.scene_description)
        print()

        # Simulate decision-making loop
        step_num = 0
        max_steps = 10

        while not result.observation.done and step_num < max_steps:
            step_num += 1

            print(f"\n--- Step {step_num} ---")

            # Simple decision logic based on scene
            if result.observation.speed_kmh > 30.0 and len(result.observation.nearby_actors) > 0:
                # Pedestrians ahead at high speed - emergency stop!
                print("Decision: EMERGENCY STOP (pedestrians detected)")
                action = CarlaAction(action_type="emergency_stop")
            else:
                # Just observe
                print("Decision: Observe")
                action = CarlaAction(action_type="observe")

            # Execute action
            result = env.step(action)

            # Show result
            print(f"Speed: {result.observation.speed_kmh:.1f} km/h")
            print(f"Collision: {result.observation.collision_detected}")

            if result.observation.collision_detected:
                print(f"⚠️  COLLISION with {result.observation.collided_with}!")

            # Small delay for readability
            time.sleep(0.5)

        # Episode ended
        print("\n" + "=" * 60)
        print("EPISODE ENDED")
        print("=" * 60)
        print(f"Reason: {result.observation.done_reason}")

        # Get final state
        state = env.state()
        print(f"\nFinal Statistics:")
        print(f"  Steps: {state.step_count}")
        print(f"  Total reward: {state.total_reward:.2f}")
        print(f"  Collisions: {len(state.collisions)}")
        print(f"  Simulation time: {state.simulation_time:.2f}s")

        if len(state.collisions) == 0:
            print("\n✅ SUCCESS: No collisions!")
        else:
            print(f"\n❌ FAILURE: {len(state.collisions)} collision(s)")

    finally:
        # Clean up
        env.close()
        print("\nEnvironment closed.")


def run_navigation_scenario():
    """Run a maze navigation scenario."""

    print("=" * 60)
    print("CARLA Environment - Maze Navigation Example")
    print("=" * 60)
    print()
    print("Note: This requires the server to be running with CARLA_SCENARIO=maze_navigation")
    print("For this example, we'll stick with the trolley scenario.")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CARLA environment examples")
    parser.add_argument(
        "--scenario",
        choices=["trolley", "navigation"],
        default="trolley",
        help="Scenario to run"
    )

    args = parser.parse_args()

    if args.scenario == "trolley":
        run_trolley_scenario()
    elif args.scenario == "navigation":
        run_navigation_scenario()
