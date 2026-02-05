#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quickstart example for the generic TextArena environment."""
from __future__ import annotations

from textarena_env import TextArenaEnv, TextArenaAction


def main() -> None:
    print("=" * 60)
    print("💬 TextArena Hello World - Wordle-v0")
    print("=" * 60)

    # TODO: move to openenv org
    env = TextArenaEnv(base_url="https://burtenshaw-wordle.hf.space")

    try:
        print("\n📍 Resetting environment...")
        result = env.reset()
        print(f"   Prompt:\n{result.observation.prompt}\n")

        # Wordle guesses - common starting words
        guesses = ["[crane]", "[slate]", "[audio]", "[pride]", "[money]", "[ghost]"]

        for step, guess in enumerate(guesses):
            print(f"🎯 Step {step + 1}: sending guess {guess}")
            result = env.step(TextArenaAction(message=guess))

            # Show the feedback
            for message in result.observation.messages:
                # Extract just the feedback part
                content = message.content
                if "Feedback:" in content:
                    feedback_part = content.split("Feedback:")[-1].strip()
                    print(f"   Feedback:\n{feedback_part}")

            if result.done:
                if result.reward and result.reward > 0:
                    print("\n🎉 You won!")
                break

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
        print("\nMake sure the server is running:")
        print("  cd envs/textarena_env && source .venv/bin/activate")
        print("  python -m uvicorn server.app:app --reload")

    finally:
        env.close()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()
