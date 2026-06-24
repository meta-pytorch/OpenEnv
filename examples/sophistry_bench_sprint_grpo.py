# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openenv[core]",
#     "trl",
#     "datasets",
#     "torch",
#     "transformers",
# ]
# ///

"""Train a policy on `sophistry_bench_sprint_env` with TRL's GRPOTrainer.

The env is single-step (`reset()` issues an advocacy task, one `step()` scores
it and ends the episode), so this is a plain prompt -> completion -> reward
GRPO setup: no `environment_factory`/tool-calling is needed (contrast with the
multi-turn Wordle GRPO tutorial). `reset(seed=i)` deterministically replays task
`i`, which is what lets the reward function re-derive a sampled completion's task
without keeping per-prompt server state around.

Uses `GenericEnvClient` (dict actions/observations) rather than the env's typed
client, so this script only depends on `openenv[core]` from PyPI -- no local
package install, which also makes it runnable as a standalone `uv` script,
including via Hugging Face Jobs:

    hf jobs uv run examples/sophistry_bench_sprint_grpo.py --flavor a10g-small \
        --secrets HF_TOKEN -- --push-to-hub --out your-username/sophistry-grpo

`make_sync_client()` (below) clones the Space's source with `git` and runs it
locally via `uv run` (`UVProvider`, the same mechanism behind
`EnvClient.from_env(..., use_docker=False)`) -- no Docker needed, and (unlike
hitting the hosted Space's public URL directly) not subject to the Space's
request quota. This needs the project_path git-clone fix from
https://github.com/huggingface/OpenEnv/pull/854; on an `openenv` release
without that fix, this hangs until the 60s readiness timeout (override the
`openenv[core]` dependency above with a git ref of that PR/branch until it's
released). `app=` is passed explicitly because this env's pyproject.toml
remaps its package dir (`server` -> `sophistry_bench_sprint_env.server`), which
doesn't match the framework's default `app="server.app:app"`. The provider is
built directly rather than via `from_env()` to avoid a sync/async event-loop
mismatch -- see the docstring on `make_sync_client()`.

Run locally:
    python examples/sophistry_bench_sprint_grpo.py --n-episodes 64 --steps 50
    # Add --push-to-hub --out your-username/sophistry-grpo to publish the
    # fine-tuned checkpoint to the Hugging Face Hub (requires `huggingface-cli login`).
"""

from __future__ import annotations

import argparse

from datasets import Dataset
from openenv import GenericEnvClient
from openenv.core.containers.runtime.uv_provider import UVProvider
from trl import GRPOConfig, GRPOTrainer

SPACE_REPO_ID = "openenv-community/sophistry_bench_sprint_env"


def build_dataset(client, n_episodes: int) -> Dataset:
    """Walk `reset(seed=i)` for i in [0, n_episodes) to get a fixed, replayable
    set of advocacy tasks. Each row carries the `seed` needed to re-derive the
    same task later, in the reward function."""
    rows = []
    for i in range(n_episodes):
        obs = client.reset(seed=i).observation
        rows.append(
            {
                "prompt": [{"role": "user", "content": obs["prompt"]}],
                "seed": i,
                "item_id": obs["item_id"],
            }
        )
    return Dataset.from_list(rows)


def make_reward_func(client):
    """`reward_funcs` callables receive the batch's `completions` plus any other
    dataset columns (here, `seed`) as keyword args. Re-running `reset(seed=...)`
    before each `step(...)` recreates the exact task the completion was sampled
    for -- the server is single-session/non-concurrent, so this must run
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
            result = client.step({"text": text})
            rewards.append(result.reward)
        return rewards

    return reward_func


def make_sync_client():
    """Build a connected `SyncEnvClient`, without going through the async
    `EnvClient.from_env()` classmethod.

    `from_env` ends with `await client.connect()`, binding the websocket to
    whichever event loop runs that coroutine. `GenericEnvClient.sync()` then
    drives all *later* calls on a second, separate background-thread loop --
    so a client connected via `asyncio.run(from_env(...))` and then wrapped in
    `.sync()` ends up with its websocket attached to a loop that's already
    closed by the time training starts. Constructing the provider directly
    (its `start()`/`wait_for_ready()` are plain sync calls, no event loop
    involved) and connecting only through the sync wrapper's own loop avoids
    the mismatch entirely.
    """
    provider = UVProvider(
        project_path=f"git+https://huggingface.co/spaces/{SPACE_REPO_ID}",
        app="sophistry_bench_sprint_env.server.app:app",
        # The default 60s readiness timeout can be too tight for a cold clone
        # + dependency install of the env project (e.g. sophistry-bench-sprint
        # pulls a QuALITY data file); give it more room.
        context_timeout_s=180.0,
    )
    base_url = provider.start()
    provider.wait_for_ready()

    client = GenericEnvClient(base_url=base_url, provider=provider)
    return client.sync()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-episodes", type=int, default=64, help="Dataset size.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--out", default="sophistry-grpo-Qwen2.5-0.5B")
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the fine-tuned model to the Hugging Face Hub under --out as the repo id.",
    )
    args = ap.parse_args()

    with make_sync_client() as client:
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
