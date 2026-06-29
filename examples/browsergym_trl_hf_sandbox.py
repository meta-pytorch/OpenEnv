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

# /// script
# dependencies = [
#     "trl[vllm,peft]",
#     "trackio",
#     "kernels",
#     "openenv-browsergym-env @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""Run TRL GRPO against BrowserGym deployed through HF Sandbox."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

from browsergym_env import BrowserGymAction, BrowserGymEnv
from datasets import Dataset
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from openenv.core.sync_client import SyncEnvClient
from trl import GRPOConfig, GRPOTrainer


MODEL_ID = "Qwen/Qwen3-0.6B"
SANDBOX_IMAGE = "hf.co/spaces/openenv/browsergym_env"
SANDBOX_FLAVOR = "cpu-basic"
BENCHMARK = "miniwob"
TASK_NAME = "click-test"
DATASET_PROMPT = "Complete the web task successfully."
DATASET_SIZE = 1000
MAX_STEPS = 10
MAX_COMPLETION_LENGTH = 1024
NUM_GENERATIONS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-6
GRADIENT_ACCUMULATION_STEPS = 32
PER_DEVICE_BATCH_SIZE = 1
VLLM_MODE = "colocate"
VLLM_SERVER_URL = "http://localhost:8001"

BROWSERGYM_ENV: SyncEnvClient[BrowserGymAction, object, object] | None = None

SYSTEM_PROMPT = """You control a web browser to complete tasks.

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means the element has bid='13'.

Use the available tools to interact with the page:
- click: Click an element by its bid
- fill: Fill an input field with text
- send_keys: Send keyboard input
- scroll: Scroll the page
- noop: Do nothing

Complete the given task as efficiently as possible."""


class BrowserGymLLMEnv:
    def __init__(self):
        if BROWSERGYM_ENV is None:
            raise RuntimeError("BrowserGym environment has not been started.")
        self.client = BROWSERGYM_ENV
        self.reward = 0.0
        self._done = False
        self._step_count = 0

    def reset(self, **kwargs) -> str:
        self.reward = 0.0
        self._done = False
        self._step_count = 0
        result = self.client.reset(task_name=TASK_NAME)
        self._done = result.done
        return self._format_observation(result.observation)

    def click(self, bid: str) -> str:
        return self._do_action(f"click({bid!r})")

    def fill(self, bid: str, text: str) -> str:
        return self._do_action(f"fill({bid!r}, {text!r})")

    def send_keys(self, text: str) -> str:
        return self._do_action(f"send_keys({text!r})")

    def scroll(self, direction: str) -> str:
        return self._do_action(f"scroll({direction!r})")

    def noop(self) -> str:
        return self._do_action("noop()")

    def _do_action(self, action_str: str) -> str:
        if self._done:
            raise ValueError("Episode is done.")

        self._step_count += 1
        result = self.client.step(BrowserGymAction(action_str=action_str))
        observation = result.observation
        step_reward = float(result.reward or 0.0)
        self._done = result.done

        if self._done and step_reward > 0:
            self.reward = 1.0
        elif self._done:
            self.reward = 0.0
        else:
            self.reward = step_reward

        if self._step_count >= MAX_STEPS:
            self._done = True

        return self._format_observation(observation)

    def _format_observation(self, observation) -> str:
        parts = []
        if observation.goal:
            parts.append(f"Goal: {observation.goal}")
        if observation.last_action_error and observation.error:
            parts.append(f"Error: {observation.error}")
        if observation.axtree_txt:
            axtree = observation.axtree_txt
            if len(axtree) > 2000:
                axtree = axtree[:2000] + "..."
            parts.append(f"Page structure:\n{axtree}")
        return "\n\n".join(parts) if parts else "No observation available."


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def reward_completion(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


def build_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": DATASET_PROMPT},
                ]
            ]
            * DATASET_SIZE
        }
    )


def main() -> None:
    global BROWSERGYM_ENV

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / f"browsergym-grpo-{sanitize_name(MODEL_ID)}-{timestamp}"
    provider = HFSandboxProvider(flavor=SANDBOX_FLAVOR)

    try:
        base_url = provider.start_container(
            SANDBOX_IMAGE,
            env_vars={
                "BROWSERGYM_BENCHMARK": BENCHMARK,
                "BROWSERGYM_TASK_NAME": TASK_NAME,
                "BROWSERGYM_HEADLESS": "true",
                "BROWSERGYM_VIEWPORT_WIDTH": "332",
                "BROWSERGYM_VIEWPORT_HEIGHT": "214",
                "MINIWOB_URL": "file:///app/miniwob-plusplus/miniwob/html/miniwob/",
            },
        )
        print(f"HF Sandbox BrowserGym URL: {base_url}")
        provider.wait_for_ready(base_url, timeout_s=300.0)

        BROWSERGYM_ENV = BrowserGymEnv(
            base_url=base_url,
            message_timeout_s=120.0,
            max_message_size_mb=100.0,
            provider=provider,
        ).sync()
        BROWSERGYM_ENV.connect()

        grpo_config = GRPOConfig(
            use_vllm=True,
            vllm_mode=VLLM_MODE,
            vllm_server_base_url=VLLM_SERVER_URL if VLLM_MODE == "server" else None,
            vllm_gpu_memory_utilization=0.4,
            output_dir=str(output_dir),
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            num_generations=NUM_GENERATIONS,
            generation_batch_size=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            report_to="trackio",
            trackio_space_id=f"browsergym-grpo-{sanitize_name(MODEL_ID)}-{timestamp}",
            chat_template_kwargs={"enable_thinking": False},
        )

        trainer = GRPOTrainer(
            model=MODEL_ID,
            reward_funcs=[reward_completion],
            train_dataset=build_dataset(),
            args=grpo_config,
            environment_factory=BrowserGymLLMEnv,
        )

        print("=" * 80)
        print("Starting GRPO training with BrowserGym on HF Sandbox")
        print(f"Benchmark: {BENCHMARK}")
        print(f"Task: {TASK_NAME}")
        print(f"Model: {MODEL_ID}")
        print(f"Environment image: {SANDBOX_IMAGE}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        trainer.train()
    finally:
        if BROWSERGYM_ENV is not None:
            BROWSERGYM_ENV.close()
        else:
            provider.stop_container()
        BROWSERGYM_ENV = None


if __name__ == "__main__":
    main()
