#!/usr/bin/env python3
"""
Example: Random Agent Playing NetHack via OpenEnv

This script demonstrates how to use the NLE environment through OpenEnv's
HTTP interface. It runs a random agent for a few episodes.

Prerequisites:
    1. Build the Docker image:
       cd src/envs/nle_env/server
       docker build -t nle-env:latest .

    2. Run this script:
       python examples/nle_random_agent.py
"""

import random
import time

# Add src to path if running directly
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from envs.nle_env import NLEEnv, NLEAction


def print_stats(observation):
    """Print human-readable stats from observation."""
    if observation.blstats is None:
        return

    blstats = observation.blstats
    # BLstats indices from NLE documentation
    print(f"  HP: {blstats[10]}/{blstats[11]}")
    print(f"  XP Level: {blstats[18]}")
    print(f"  Gold: {blstats[13]}")
    print(f"  Dungeon Level: {blstats[12]}")


def main():
    print("=" * 70)
    print("NLE Random Agent Example")
    print("=" * 70)

    # Start environment (automatically launches Docker container)
    print("\n[1/3] Starting NLE environment...")
    print("(This may take a moment if container needs to start)")

    env = NLEEnv.from_docker_image(
        "nle-env:latest",
        # Optional: customize container
        # env_vars={"NLE_MAX_STEPS": "1000"}
    )

    print("✓ Environment connected!")

    # Run a few episodes
    num_episodes = 3
    max_steps_per_episode = 100

    print(f"\n[2/3] Running {num_episodes} episodes...")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # Reset environment
        result = env.reset()
        print("Environment reset")
        print_stats(result.observation)

        episode_reward = 0
        steps = 0

        # Play episode
        for step in range(max_steps_per_episode):
            # Random action (0-112)
            action = NLEAction(action_id=random.randint(0, 112))

            # Take step
            result = env.step(action)

            episode_reward += result.reward or 0
            steps += 1

            # Print occasional updates
            if step % 20 == 0:
                print(f"  Step {step}: reward={episode_reward:.1f}")

            # Check if done
            if result.done:
                state = env.state()
                print(f"\nEpisode ended after {steps} steps!")
                print(f"  Total reward: {episode_reward:.1f}")
                print(f"  End status: {state.end_status}")
                print(f"  Final stats:")
                print_stats(result.observation)
                break
        else:
            print(f"\nReached max steps ({max_steps_per_episode})")
            print(f"  Total reward: {episode_reward:.1f}")

        time.sleep(0.5)  # Brief pause between episodes

    # Cleanup
    print("\n[3/3] Cleaning up...")
    env.close()
    print("✓ Environment closed")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
