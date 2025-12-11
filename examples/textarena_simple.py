#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quickstart example for the generic TextArena environment."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project src/ to import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.textarena_env import TextArenaEnv, TextArenaAction


def main() -> None:
    
    print("=" * 60)
    print("💬 TextArena Hello World - GuessTheNumber-v0")
    print("=" * 60)

    env = TextArenaEnv.from_docker_image(
        "textarena-env:latest",
        env_vars={
            "TEXTARENA_ENV_ID": "GuessTheNumber-v0",
            "TEXTARENA_NUM_PLAYERS": "1",
        },
        ports={8000: 8000},
    )

    try:
        print("\n📍 Resetting environment...")
        result = env.reset()
        print(f"   Prompt:\n{result.observation.prompt}\n")

        # Simple heuristic: if prompt mentions a range, start with midpoint
        guess = "[10]"

        for step in range(5):
            print(f"🎯 Step {step + 1}: sending guess {guess}")
            result = env.step(TextArenaAction(message=guess))

            for message in result.observation.messages:
                print(f"   [{message.category}] {message.content}")

            if result.done:
                break

            # Basic update: look for 'higher' or 'lower' hints
            feedback = " ".join(msg.content for msg in result.observation.messages)
            if "higher" in feedback:
                guess = "[15]"
            elif "lower" in feedback:
                guess = "[5]"
            else:
                guess = "[10]"

        print("\n✅ Episode finished!")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        state = env.state()
        print("\n📊 Server State Snapshot:")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Env ID: {state.env_id}")

    except Exception as exc:  # pragma: no cover - demonstration script
        print(f"\n❌ Error: {exc}")
        print("\nMake sure you have built the Docker image first:")
        print("  docker build -f envs/textarena_env/server/Dockerfile -t textarena-env:latest .")
        print("\nAlternatively run the server manually:")
        print("  python -m envs.textarena_env.server.app")

    finally:
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

