#!/usr/bin/env python3
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke-check BrowserGym through the HF sandbox provider."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from browsergym_env import BrowserGymAction, BrowserGymEnv
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider


SANDBOX_IMAGE = "hf.co/spaces/openenv/browsergym_env"
TASK_NAME = "click-test"


def make_provider() -> HFSandboxProvider:
    return HFSandboxProvider(
        image=SANDBOX_IMAGE,
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": TASK_NAME,
            "BROWSERGYM_HEADLESS": "true",
            "BROWSERGYM_VIEWPORT_WIDTH": "332",
            "BROWSERGYM_VIEWPORT_HEIGHT": "214",
            "MINIWOB_URL": "file:///app/miniwob-plusplus/miniwob/html/miniwob/",
        },
    )


def first_button_bid(axtree: str) -> str:
    match = re.search(r"\[(\d+)\]\s+button", axtree)
    if match is None:
        raise RuntimeError(f"No button found in accessibility tree:\n{axtree}")
    return match.group(1)


def main() -> None:
    with BrowserGymEnv(
        message_timeout_s=120.0,
        max_message_size_mb=100.0,
        provider=make_provider(),
    ).sync() as env:
        reset = env.reset()
        bid = first_button_bid(reset.observation.axtree_txt or "")
        action = f"click({bid!r})"
        result = env.step(BrowserGymAction(action_str=action))

    print(f"goal: {reset.observation.goal}")
    print(f"action: {action}")
    print(f"reward: {result.reward}")
    print(f"done: {result.done}")
    if result.reward != 1.0 or not result.done:
        raise RuntimeError("BrowserGym HF sandbox smoke check failed.")


if __name__ == "__main__":
    main()
