#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example of using Maze environment with OpenEnv.

This demonstrates:
1. Connecting to the Maze environment server
2. Resetting the environment
3. Taking actions
4. Observing rewards
5. Inspecting environment state

Usage:
    python examples/maze_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np
from envs.maze_env import MazeEnv, MazeAction


def main():
    print("🧩 Simple Maze Environment Example")
    print("=" * 60)

    # Connect to environment server
    # Ensure server is running: python -m envs.maze_env.server.app
    env = MazeEnv(base_url="http://localhost:8000")
    maze = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
    try:
        # Reset environment
        print("\n📍 Resetting environment...")
        result = env.reset()

        print(f"   Initial position: {result.observation.position}")
        print(f"   Legal actions: {result.observation.legal_actions}")

        # Run one episode
        print("\n🚶 Navigating through maze...")
        step = 0
        total_reward = 0

        while not result.done and step < 20:
            # Choose random legal action
            print(f"   Current position: {result.observation.position}")
            print(f"   Legal actions: {result.observation.legal_actions}")
            env.render_ascii_maze(maze,result.observation.position,[0,0],[maze.shape[0],maze.shape[1]])
            action_id = result.observation.legal_actions[step % len(result.observation.legal_actions)]
            # Take action
            result = env.step(MazeAction(action=action_id))

            reward = result.reward or 0
            total_reward += reward

            print(f"   Step {step + 1}: action={action_id}, pos={result.observation.position}, reward={reward:.2f}, done={result.done}")
            step += 1
            print("-----------------------------------------------------")

        print(f"\n✅ Episode finished!")
        print(f"   Total steps: {step}")
        print(f"   Total reward: {total_reward}")

        # Get environment state
        state = env.state()
        print(f"\n📊 Environment State:")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Done: {state.done}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m envs.maze_env.server.app")
        print("\nOr start with Docker:")
        print("  docker run -p 8000:8000 maze-env:latest")

    finally:
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()
