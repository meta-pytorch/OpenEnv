#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usage of the Doom Environment.

This script demonstrates how to use the Doom environment with OpenEnv.
It can be run in two modes:
1. With Docker: Uses the Docker image to run the environment
2. Local: Directly uses the DoomEnvironment class (requires ViZDoom installed)
"""

import argparse
import numpy as np


def run_with_docker():
    """Run Doom environment using Docker container."""
    from doom_env import DoomAction, DoomEnv

    print("Starting Doom environment with Docker...")
    try:
        # Create environment from Docker image
        env = DoomEnv.from_docker_image("doom-env:latest")

        # Reset to start a new episode
        result = env.reset()
        print(f"✓ Environment reset")
        print(f"  Screen shape: {result.observation.screen_shape}")
        print(f"  Available actions: {result.observation.available_actions}")
        print(f"  Game variables: {result.observation.game_variables}")

        # Run for a few steps
        print("\nRunning episode...")
        for i in range(20):
            # Take a random action from available actions
            available_actions = result.observation.available_actions
            if available_actions:
                action_id = np.random.choice(available_actions)
            else:
                action_id = 0

            result = env.step(DoomAction(action_id=action_id))

            print(f"Step {i+1}:")
            print(f"  Action: {action_id}")
            print(f"  Reward: {result.reward}")
            print(f"  Done: {result.observation.done}")

            if result.observation.done:
                print("\n✓ Episode finished!")
                break

    finally:
        print("\nCleaning up...")
        env.close()
        print("✓ Environment closed")


def run_local():
    """Run Doom environment locally without Docker."""
    try:
        from server.doom_env_environment import DoomEnvironment
        from models import DoomAction
    except ImportError as e:
        print(f"Error: Could not import environment components: {e}")
        print("Make sure ViZDoom is installed: pip install vizdoom")
        return

    print("Starting Doom environment locally...")
    try:
        # Create environment
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_160X120",
            screen_format="RGB24",
            window_visible=False,
            use_discrete_actions=True,
        )

        # Reset to start a new episode
        obs = env.reset()
        print(f"✓ Environment reset")
        print(f"  Screen shape: {obs.screen_shape}")
        print(f"  Available actions: {obs.available_actions}")
        print(f"  Game variables: {obs.game_variables}")

        # Run for a few steps
        print("\nRunning episode...")
        for i in range(20):
            # Take a random action
            if obs.available_actions:
                action_id = np.random.choice(obs.available_actions)
            else:
                action_id = 0

            obs = env.step(DoomAction(action_id=action_id))

            print(f"Step {i+1}:")
            print(f"  Action: {action_id}")
            print(f"  Reward: {obs.reward}")
            print(f"  Done: {obs.done}")

            if obs.done:
                print("\n✓ Episode finished!")
                break

        # Visualize a frame
        if not obs.done:
            screen = np.array(obs.screen_buffer).reshape(obs.screen_shape)
            print(f"\nScreen statistics:")
            print(f"  Shape: {screen.shape}")
            print(f"  Dtype: {screen.dtype}")
            print(f"  Min: {screen.min()}, Max: {screen.max()}")

    except Exception as e:
        print(f"Error running environment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
            print("✓ Environment closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Doom environment example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Docker
  python example.py --docker

  # Run locally
  python example.py --local

  # Run with default (Docker)
  python example.py
        """
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        default=True,
        help="Run with Docker container (default)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally without Docker"
    )

    args = parser.parse_args()

    if args.local:
        run_local()
    else:
        run_with_docker()


if __name__ == "__main__":
    main()
