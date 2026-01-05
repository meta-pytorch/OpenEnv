#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple example of using OpenSpiel environment with OpenEnv.

This demonstrates the basic workflow:
1. Connect to environment
2. Reset
3. Take actions
4. Observe rewards
5. Close

Usage:
    python examples/openspiel_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openspiel_env import OpenSpielEnv, OpenSpielAction


def main():
    print("🎯 Simple OpenSpiel Example - Catch Game")
    print("=" * 60)

    # Connect to environment server
    # Make sure server is running: python -m envs.openspiel_env.server.app
    env = OpenSpielEnv(base_url="http://localhost:8000")

    try:
        # Reset environment
        print("\n📍 Resetting environment...")
        result = env.reset()

        print(f"   Initial observation shape: {len(result.observation.info_state)}")
        print(f"   Legal actions: {result.observation.legal_actions}")
        print(f"   Game phase: {result.observation.game_phase}")

        # Run one episode
        print("\n🎮 Playing episode...")
        step = 0
        total_reward = 0

        while not result.done and step < 20:
            # Choose first legal action (you can use any policy here)
            action_id = result.observation.legal_actions[0]

            # Take action
            result = env.step(OpenSpielAction(action_id=action_id, game_name="catch"))

            # Track reward
            reward = result.reward or 0
            total_reward += reward

            print(f"   Step {step + 1}: action={action_id}, reward={reward:.2f}, done={result.done}")
            step += 1

        # Episode finished
        print(f"\n✅ Episode finished!")
        print(f"   Total steps: {step}")
        print(f"   Total reward: {total_reward}")
        print(f"   Result: {'Ball caught! 🎉' if total_reward > 0 else 'Ball missed 😢'}")

        # Get environment state
        state = env.state()
        print(f"\n📊 Environment State:")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Game: {state.game_name}")
        print(f"   Num players: {state.num_players}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m envs.openspiel_env.server.app")
        print("\nOr start with Docker:")
        print("  docker run -p 8000:8000 openspiel-env:latest")

    finally:
        # Always close the environment
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()
