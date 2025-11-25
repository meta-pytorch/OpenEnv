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

Both modes support rendering if the appropriate libraries are installed.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_with_docker(render: bool = False, num_steps: int = 100):
    """Run Doom environment using Docker container."""
    from envs.doom_env import DoomAction, DoomEnv

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

        if render:
            print(f"\n✓ Rendering enabled")
            print(f"  Note: Rendering uses the observation from the server")
            print(
                f"  For best performance, consider using local mode with window_visible=True"
            )
            print(f"  Alternative: Use the web interface at http://localhost:8000/web")
            print(
                f"  Make sure opencv-python or matplotlib is installed locally for rendering"
            )

        # Run for a few steps
        print(f"\nRunning episode for {num_steps} steps...")
        for i in range(num_steps):
            # Take a random action from available actions
            available_actions = result.observation.available_actions
            if available_actions:
                action_id = int(
                    np.random.choice(available_actions)
                )  # Convert to Python int
            else:
                action_id = 0

            result = env.step(DoomAction(action_id=action_id))

            # Render if requested
            if render:
                env.render()

            if i % 10 == 0 or result.observation.done:
                print(f"Step {i+1}:")
                print(f"  Action: {action_id}")
                print(f"  Reward: {result.reward}")
                print(f"  Done: {result.observation.done}")
                if result.observation.game_variables:
                    print(f"  Game vars: {result.observation.game_variables}")

            if result.observation.done:
                print(f"\n✓ Episode finished at step {i+1}!")
                break

            # Small delay for rendering to be visible
            if render:
                time.sleep(0.03)

    finally:
        print("\nCleaning up...")
        env.close()
        print("✓ Environment closed")


def run_local(render: bool = False, num_steps: int = 100):
    """Run Doom environment locally without Docker."""
    try:
        from envs.doom_env.models import DoomAction
        from envs.doom_env.server.doom_env_environment import DoomEnvironment
    except ImportError as e:
        print(f"Error: Could not import environment components: {e}")
        print("Make sure ViZDoom is installed: pip install vizdoom")
        return

    print("Starting Doom environment locally...")
    try:
        # Create environment
        # Note: When using local mode, you can enable window_visible=True
        # for native ViZDoom rendering (most efficient)
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_320X240",  # Higher resolution for better visibility
            screen_format="RGB24",
            window_visible=render,  # Use native ViZDoom window if rendering
            use_discrete_actions=True,
        )

        # Reset to start a new episode
        obs = env.reset()
        print(f"✓ Environment reset")
        print(f"  Screen shape: {obs.screen_shape}")
        print(f"  Available actions: {obs.available_actions}")
        print(f"  Game variables: {obs.game_variables}")

        if render:
            print(f"\n✓ Rendering enabled")
            if env.window_visible:
                print(f"  Using native ViZDoom window (most efficient)")
            else:
                print(f"  Using Python rendering (cv2/matplotlib)")

        # Run for a few steps
        print(f"\nRunning episode for {num_steps} steps...")
        for i in range(num_steps):
            # Take a random action
            if obs.available_actions:
                action_id = int(
                    np.random.choice(obs.available_actions)
                )  # Convert to Python int
            else:
                action_id = 0

            obs = env.step(DoomAction(action_id=action_id))

            # Render if requested and not using native window
            if render and not env.window_visible:
                env.render()

            if i % 10 == 0 or obs.done:
                print(f"Step {i+1}:")
                print(f"  Action: {action_id}")
                print(f"  Reward: {obs.reward}")
                print(f"  Done: {obs.done}")
                if obs.game_variables:
                    print(f"  Game vars: {obs.game_variables}")

            if obs.done:
                print(f"\n✓ Episode finished at step {i+1}!")
                break

            # Small delay for rendering to be visible
            if render:
                time.sleep(0.03)

        # Visualize final frame statistics
        if not obs.done:
            screen = np.array(obs.screen_buffer).reshape(obs.screen_shape)
            print(f"\nFinal screen statistics:")
            print(f"  Shape: {screen.shape}")
            print(f"  Dtype: {screen.dtype}")
            print(f"  Min: {screen.min()}, Max: {screen.max()}")

    except Exception as e:
        print(f"Error running environment: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "env" in locals():
            env.close()
            print("✓ Environment closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Doom environment example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Docker (default)
  python example.py

  # Run with Docker and rendering
  python example.py --render

  # Run locally without Docker
  python example.py --local

  # Run locally with rendering (uses native ViZDoom window)
  python example.py --local --render

  # Run for more steps
  python example.py --local --render --steps 300

Note: Rendering requires opencv-python or matplotlib:
  pip install opencv-python
  # or
  pip install matplotlib
        """,
    )
    parser.add_argument(
        "--docker", action="store_true", default=False, help="Run with Docker container"
    )
    parser.add_argument(
        "--local", action="store_true", help="Run locally without Docker"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (shows game window or visualization)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps to run (default: 100)",
    )

    args = parser.parse_args()

    # Default to docker if neither specified
    if not args.local and not args.docker:
        args.docker = True

    if args.local:
        run_local(render=args.render, num_steps=args.steps)
    else:
        run_with_docker(render=args.render, num_steps=args.steps)


if __name__ == "__main__":
    main()
