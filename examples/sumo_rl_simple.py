#!/usr/bin/env python3
"""
Simple example demonstrating SUMO-RL Environment usage.

This example shows how to:
1. Connect to a SUMO traffic signal control environment
2. Reset the environment
3. Take actions (select traffic light phases)
4. Process observations and rewards

Usage:
    # Option 1: Start the server manually
    python -m envs.sumo_rl_env.server.app
    # Then run: python examples/sumo_rl_simple.py

    # Option 2: Use Docker
    docker run -p 8000:8000 sumo-rl-env:latest
    # Then run: python examples/sumo_rl_simple.py
"""

import numpy as np

from sumo_rl_env import SumoAction, SumoRLEnv


def main():
    """Run a simple SUMO traffic control episode."""
    # Connect to the SUMO environment server
    print("Connecting to SUMO-RL environment...")
    env = SumoRLEnv(base_url="http://localhost:8000")

    try:
        # Reset the environment
        print("\nResetting environment...")
        result = env.reset()
        print(f"Observation shape: {result.observation.observation_shape}")
        print(f"Available actions: {result.observation.action_mask}")
        print(f"Number of green phases: {len(result.observation.action_mask)}")

        # Get initial state
        state = env.state()
        print(f"\nSimulation configuration:")
        print(f"  Network: {state.net_file}")
        print(f"  Duration: {state.num_seconds} seconds")
        print(f"  Delta time: {state.delta_time} seconds")
        print(f"  Reward function: {state.reward_fn}")

        # Run a few steps with random policy
        print("\nRunning traffic control with random policy...")
        episode_reward = 0
        steps = 0
        max_steps = 100

        for step in range(max_steps):
            # Random policy: select random green phase
            action_id = np.random.choice(result.observation.action_mask)

            # Take action
            result = env.step(SumoAction(phase_id=int(action_id)))

            episode_reward += result.reward or 0
            steps += 1

            # Print progress every 10 steps
            if step % 10 == 0:
                state = env.state()
                print(
                    f"Step {step:3d}: "
                    f"phase={action_id}, "
                    f"reward={result.reward:6.2f}, "
                    f"vehicles={state.total_vehicles:3d}, "
                    f"waiting={state.mean_waiting_time:6.2f}s, "
                    f"speed={state.mean_speed:5.2f}m/s"
                )

            if result.done:
                print(f"\nEpisode finished after {steps} steps!")
                break

        # Final statistics
        print(f"\n{'='*60}")
        print(f"Episode Summary:")
        print(f"  Total steps: {steps}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Average reward: {episode_reward/steps:.2f}")

        # Get final state
        state = env.state()
        print(f"\nFinal State:")
        print(f"  Simulation time: {state.sim_time:.0f} seconds")
        print(f"  Total vehicles: {state.total_vehicles}")
        print(f"  Total waiting time: {state.total_waiting_time:.2f} seconds")
        print(f"  Mean waiting time: {state.mean_waiting_time:.2f} seconds")
        print(f"  Mean speed: {state.mean_speed:.2f} m/s")
        print(f"{'='*60}")

    finally:
        # Cleanup
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
