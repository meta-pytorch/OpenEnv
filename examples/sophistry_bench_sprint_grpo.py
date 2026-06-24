# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train a policy on `sophistry_bench_sprint_env` with TRL's GRPOTrainer.

The env is single-step (`reset()` issues an advocacy task, one `step_text(...)`
scores it and ends the episode), so this is a plain prompt -> completion -> reward
GRPO setup: no `environment_factory`/tool-calling is needed (contrast with the
multi-turn Wordle GRPO tutorial). `reset(seed=i)` deterministically replays task
`i`, which is what lets the reward function re-derive a sampled completion's task
without keeping per-prompt server state around.

Requires Docker (the env is pulled from the Hugging Face Space and run locally,
which both avoids the hosted Space's concurrency limits and keeps the held-out
`correctness_reward` off the wire by default -- see envs/sophistry_bench_sprint_env/README.md).

Install:
    pip install "trl[vllm]" datasets torch
    pip install -e envs/sophistry_bench_sprint_env

Run:
    python examples/sophistry_bench_sprint_grpo.py --n-episodes 64 --steps 50
    # Add --push-to-hub --out your-username/sophistry-grpo to publish the
    # fine-tuned checkpoint to the Hugging Face Hub (requires `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import asyncio

from datasets import Dataset
from sophistry_bench_sprint_env import SophistryBenchSprintEnv
from trl import GRPOConfig, GRPOTrainer


def build_dataset(client, n_episodes: int) -> Dataset:
    """Walk `reset(seed=i)` for i in [0, n_episodes) to get a fixed, replayable
    set of advocacy tasks. Each row carries the `seed` needed to re-derive the
    same task later, in the reward function."""
    rows = []
    for i in range(n_episodes):
        obs = client.reset(seed=i).observation
        rows.append(
            {
                "prompt": [{"role": "user", "content": obs.prompt}],
                "seed": i,
                "item_id": obs.item_id,
            }
        )
    return Dataset.from_list(rows)


def make_reward_func(client):
    """`reward_funcs` callables receive the batch's `completions` plus any other
    dataset columns (here, `seed`) as keyword args. Re-running `reset(seed=...)`
    before each `step_text(...)` recreates the exact task the completion was
    sampled for -- the server is single-session/non-concurrent, so this must run
    sequentially against one client.
    """

    def reward_func(completions, seed, **kwargs) -> list[float]:
        rewards = []
        for completion, s in zip(completions, seed):
            client.reset(seed=s)
            text = (
                completion[-1]["content"]
                if isinstance(completion, list)
                else completion
            )
            result = client.step_text(text)
            rewards.append(result.reward)
        return rewards

    return reward_func


async def make_client() -> SophistryBenchSprintEnv:
    # Pulls and runs the published container locally via Docker rather than
    # hitting the hosted Space (recommended for training throughput).
    return await SophistryBenchSprintEnv.from_env(
        "openenv-community/sophistry_bench_sprint_env"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n-episodes", type=int, default=64, help="Dataset size.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--out", default="sophistry-grpo-Qwen3-1.7B")
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the fine-tuned model to the Hugging Face Hub under --out as the repo id.",
    )
    args = ap.parse_args()

    async_client = asyncio.run(make_client())
    client = async_client.sync()

    with client:
        dataset = build_dataset(client, args.n_episodes)
        reward_func = make_reward_func(client)

        config = GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            learning_rate=args.lr,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=512,
            log_completions=True,
            logging_steps=1,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.out if args.push_to_hub else None,
        )

        trainer = GRPOTrainer(
            model=args.model,
            reward_funcs=reward_func,
            train_dataset=dataset,
            args=config,
        )
        trainer.train()
        trainer.save_model(args.out)
        print(f"Saved fine-tuned model to {args.out}")

        if args.push_to_hub:
            trainer.push_to_hub()
            print(f"Pushed fine-tuned model to https://huggingface.co/{args.out}")


if __name__ == "__main__":
    main()
