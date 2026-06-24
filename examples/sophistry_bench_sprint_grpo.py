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
without that fix, this hangs until the readiness timeout (override the
`openenv[core]` dependency above with a git ref of that PR/branch until it's
released). `app=` is passed explicitly because this env's pyproject.toml
remaps its package dir (`server` -> `sophistry_bench_sprint_env.server`),
which doesn't match the framework's default `app="server.app:app"`.

The provider is built directly rather than via `from_env()` for two reasons,
both covered in `make_sync_client()`'s docstring: (1) `from_env()` + `.sync()`
has a sync/async event-loop mismatch (also fixed in #854 -- see that PR), and
(2) even with that fixed, this env in particular only allows **one concurrent
session** (`SUPPORTS_CONCURRENT_SESSIONS = False`), so the orphaned first
connection that the event-loop bug leaves behind would occupy that single
slot and the real connection would fail with `CAPACITY_REACHED` -- the
process-level fix doesn't have a way to cleanly close a websocket whose
event loop is already gone. Connecting only once, through the sync wrapper's
own loop, avoids creating that orphaned connection in the first place.

Run locally:
    python examples/sophistry_bench_sprint_grpo.py --n-episodes 64 --steps 50
    # Add --push-to-hub --out your-username/sophistry-grpo to publish the
    # fine-tuned checkpoint to the Hugging Face Hub (requires `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import csv
import os

from datasets import Dataset
from openenv import GenericEnvClient
from openenv.core.containers.runtime.uv_provider import UVProvider
from trl import GRPOConfig, GRPOTrainer

SPACE_REPO_ID = "openenv-community/sophistry_bench_sprint_env"


def _completion_text(completion) -> str:
    """Extract the assistant's text from a TRL completion.

    TRL passes either a list of chat messages (use the last one's content) or
    a raw string, depending on whether the model/dataset use chat templating.
    """
    if isinstance(completion, list):
        if not completion or not isinstance(completion[-1], dict):
            raise ValueError(f"Unexpected completion shape from TRL: {completion!r}")
        return completion[-1]["content"]
    if isinstance(completion, str):
        return completion
    raise ValueError(f"Unexpected completion type from TRL: {type(completion)!r}")


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


def make_reward_func(client, metrics_log):
    """`reward_funcs` callables receive the batch's `completions` plus any other
    dataset columns (here, `seed`) as keyword args. Re-running `reset(seed=...)`
    before each `step(...)` recreates the exact task the completion was sampled
    for -- the server is single-session/non-concurrent, so this must run
    sequentially against one client.

    `result.reward` (the weighted aggregate the trainer optimizes) is the only
    thing TRL needs back, but `result.observation["components"]` carries the
    full 7-8 reward sub-scores -- including `correctness_reward`, the hidden
    ground truth that's *not* in the optimized objective. Averaging those per
    step and appending to `metrics_log` is what lets the README compare
    "proxy reward up" against "correctness flat", the way the Prime Intellect
    run's metrics.csv does.
    """
    step = 0

    def reward_func(completions, seed, **kwargs) -> list[float]:
        nonlocal step
        assert len(completions) == len(seed), (
            f"completions/seed length mismatch: {len(completions)} vs {len(seed)} "
            "-- reward can't be paired with the task it was scored against"
        )
        rewards = []
        components = []
        for completion, s in zip(completions, seed):
            client.reset(seed=s)
            text = _completion_text(completion)
            result = client.step({"text": text})
            rewards.append(result.reward)
            components.append(result.observation.get("components") or {})

        step += 1
        keys = sorted({k for c in components for k in c})
        row = {
            "step": step,
            "reward": sum(rewards) / len(rewards),
            **{
                k: sum(c.get(k, 0.0) for c in components) / len(components)
                for k in keys
            },
        }
        metrics_log.append(row)
        print(f"[components] {row}")
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
    closed by the time training starts. The fixed `connect()` in #854 detects
    this and reconnects on the new loop rather than hanging, but it can't
    cleanly close the *old* connection first (its loop is already gone), so
    the old one is simply abandoned -- harmless for envs that allow
    concurrent sessions, but this env doesn't
    (`SUPPORTS_CONCURRENT_SESSIONS = False`), so the abandoned connection
    occupies the only session slot and the real one fails with
    `CAPACITY_REACHED`. Constructing the provider directly (its
    `start()`/`wait_for_ready()` are plain sync calls, no event loop
    involved) and connecting only through the sync wrapper's own loop avoids
    ever creating that doomed first connection.
    """
    provider = UVProvider(
        project_path=f"git+https://huggingface.co/spaces/{SPACE_REPO_ID}",
        app="sophistry_bench_sprint_env.server.app:app",
        # The default 60s readiness timeout can be too tight for a cold clone
        # + dependency install of the env project (e.g. sophistry-bench-sprint
        # pulls a QuALITY data file); give it more room.
        context_timeout_s=180.0,
        # correctness_reward (the hidden ground truth) is withheld from the
        # wire by default so a harness can't leak it to the policy. We're
        # running our own local clone, not the shared public Space, and only
        # logging this for offline metrics (never feeding it back into the
        # prompt) -- exactly the "trusted measurement code" opt-in the env's
        # own README describes.
        env_vars={"SPRINT_EXPOSE_CORRECTNESS": "1"},
    )
    base_url = provider.start()
    provider.wait_for_ready()

    client = GenericEnvClient(base_url=base_url, provider=provider)
    return client.sync()


def write_metrics_csv(metrics_log: list[dict], path: str) -> None:
    if not metrics_log:
        return
    fieldnames = sorted({k for row in metrics_log for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"Wrote per-step component metrics to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-episodes", type=int, default=64, help="Dataset size.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Total rollouts sampled per step (must be divisible by --num-generations).",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=2,
        help="Completions sampled per prompt per step.",
    )
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--out", default="sophistry-grpo-Qwen2.5-0.5B")
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the fine-tuned model to the Hugging Face Hub under --out as the repo id.",
    )
    args = ap.parse_args()

    if args.per_device_batch_size % args.num_generations != 0:
        ap.error(
            f"--per-device-batch-size ({args.per_device_batch_size}) must be "
            f"divisible by --num-generations ({args.num_generations})"
        )

    with make_sync_client() as client:
        dataset = build_dataset(client, args.n_episodes)
        metrics_log: list[dict] = []
        reward_func = make_reward_func(client, metrics_log)

        config = GRPOConfig(
            output_dir=args.out,
            max_steps=args.steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_batch_size,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            # The per-token-logprob/entropy computation materializes a
            # [batch, completion_len, vocab_size] logits tensor; with a
            # ~150K-token vocab (e.g. Qwen2.5) that dominates GPU memory, so
            # bf16 (half the bytes of fp32) matters more here than usual.
            bf16=True,
            # The prompts here embed full QuALITY passages (~1500-2500 tokens),
            # so backward-pass activation memory across all layers is the
            # other big cost on top of the long-vocab logits above; trade
            # some speed for memory by recomputing forward activations during
            # backward instead of storing them.
            gradient_checkpointing=True,
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

        # output_dir (args.out) is guaranteed to exist by save_model() above,
        # unlike an arbitrary sibling path built from args.out's basename.
        write_metrics_csv(metrics_log, os.path.join(args.out, "components.csv"))

        if args.push_to_hub:
            trainer.push_to_hub()
            print(f"Pushed fine-tuned model to https://huggingface.co/{args.out}")


if __name__ == "__main__":
    main()
