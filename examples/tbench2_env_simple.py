#!/usr/bin/env python3
"""Minimal example for the TB2 OpenEnv runner (local mode)."""

import os

from tbench2_env import Tbench2Action, Tbench2Env


def main() -> int:
    tasks_dir = os.environ.get("TB2_TASKS_DIR")
    if not tasks_dir:
        print("TB2_TASKS_DIR not set. TB2 repo will be downloaded.")

    task_id = os.environ.get("TB2_TASK_ID", "headless-terminal")
    base_url = os.environ.get("TB2_BASE_URL", "http://localhost:8000")

    env = Tbench2Env(base_url=base_url)
    result = env.reset(task_id=task_id)
    print("Instruction head:")
    print(result.observation.instruction[:200])

    result = env.step(Tbench2Action(action_type="exec", command="ls -la"))
    print("Command output:")
    print(result.observation.output[:400])

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
