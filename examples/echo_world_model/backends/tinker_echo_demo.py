# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tinker==0.22.7",
#     "torch>=2.2",
# ]
# ///

"""Train the ECHO verifier-free world-model objective with Tinker.

This is the Tinker-backed equivalent of ``../train_echo.py``. It reuses the
same deterministic terminal rollouts and their per-token role masks, but sends
the forward/backward and optimizer work to Tinker's remote LoRA trainer.

Only real ``env_output`` tokens receive cross-entropy weight. Agent actions,
context, and harness warnings remain conditioning context with zero loss weight.

Run from ``examples/echo_world_model``:

    export TINKER_API_KEY="..."
    uv run backends/tinker_echo_demo.py --steps 15
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLE_DIR))

from mini_terminal_env import TEST_TASKS, TRAIN_TASKS  # noqa: E402
from rollout import oracle_rollout  # noqa: E402
from trajectory import Trajectory, tokenize_trajectory  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def build_echo_data(
    tinker_types: Any,
    tokenizer: Any,
    trajectories: Sequence[Trajectory],
) -> list[Any]:
    """Convert role-tagged ECHO trajectories into Tinker training data.

    The causal-language-model shift is applied explicitly: input token ``t``
    predicts target token ``t + 1``, and the target's role determines its loss
    weight. Weights are normalized across the batch so Tinker's sum-reduced
    cross-entropy equals mean env-token cross-entropy.
    """
    tokenized = [tokenize_trajectory(tokenizer, traj) for traj in trajectories]
    env_token_count = sum(int(item["obs_mask"][1:].sum()) for item in tokenized)
    if env_token_count == 0:
        raise ValueError("ECHO data must contain at least one env_output target token")

    data = []
    for item in tokenized:
        token_ids = item["input_ids"].tolist()
        if len(token_ids) < 2:
            raise ValueError("ECHO trajectories must contain at least two tokens")

        target_tokens = token_ids[1:]
        weights = [
            float(is_env_target) / env_token_count
            for is_env_target in item["obs_mask"][1:].tolist()
        ]
        data.append(
            tinker_types.Datum(
                model_input=tinker_types.ModelInput.from_ints(tokens=token_ids[:-1]),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights,
                },
            )
        )
    return data


def _values(tensor: Any) -> list[float]:
    values = tensor.tolist() if hasattr(tensor, "tolist") else tensor
    return [float(value) for value in values]


def mean_env_ce(result: Any, data: Sequence[Any]) -> float:
    """Compute mean env-token CE from a Tinker forward result."""
    if len(result.loss_fn_outputs) != len(data):
        raise ValueError("Tinker returned a different number of outputs than data")

    weighted_logprob = 0.0
    total_weight = 0.0
    for output, datum in zip(result.loss_fn_outputs, data, strict=True):
        logprobs = _values(output["logprobs"])
        weights = _values(datum.loss_fn_inputs["weights"])
        if len(logprobs) != len(weights):
            raise ValueError("Tinker logprobs and ECHO weights must be aligned")
        weighted_logprob += sum(
            logprob * weight for logprob, weight in zip(logprobs, weights, strict=True)
        )
        total_weight += sum(weights)

    if total_weight <= 0:
        raise ValueError("ECHO data must have positive env-token weight")
    return -weighted_logprob / total_weight


def evaluate_env_ce(training_client: Any, data: Sequence[Any]) -> float:
    """Run a no-gradient Tinker forward pass and return env-token CE."""
    result = training_client.forward(list(data), "cross_entropy").result()
    return mean_env_ce(result, data)


def train_step(
    training_client: Any,
    tinker_types: Any,
    data: Sequence[Any],
    learning_rate: float,
) -> float:
    """Submit one pipelined Tinker forward/backward and optimizer step."""
    fwdbwd_future = training_client.forward_backward(list(data), "cross_entropy")
    optim_future = training_client.optim_step(
        tinker_types.AdamParams(learning_rate=learning_rate)
    )
    result = fwdbwd_future.result()
    optim_future.result()
    return mean_env_ce(result, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--checkpoint-name",
        help="Optionally save Tinker weights and optimizer state under this name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be at least 1")
    if not os.getenv("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is required. Create a key in the Tinker Console, "
            "export it, and run this command again."
        )

    try:
        import tinker
    except ImportError as error:
        raise SystemExit(
            "The Tinker SDK is missing. Run this script with "
            "`uv run backends/tinker_echo_demo.py`."
        ) from error

    print(f"Creating Tinker LoRA trainer for {args.model} (rank={args.rank})")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.rank,
    )
    tokenizer = training_client.get_tokenizer()

    train_data = build_echo_data(
        tinker.types,
        tokenizer,
        [oracle_rollout(task) for task in TRAIN_TASKS],
    )
    heldout_data = build_echo_data(
        tinker.types,
        tokenizer,
        [oracle_rollout(task) for task in TEST_TASKS],
    )

    before = evaluate_env_ce(training_client, heldout_data)
    print(f"Held-out env-token CE before training: {before:.3f} nats/token")

    after = before
    for step in range(1, args.steps + 1):
        train_ce = train_step(
            training_client,
            tinker.types,
            train_data,
            args.learning_rate,
        )
        if step % args.eval_every == 0 or step == args.steps:
            after = evaluate_env_ce(training_client, heldout_data)
            print(
                f"step {step:3d}  train env-CE {train_ce:.3f}  |  "
                f"held-out env-CE {after:.3f}"
            )
        else:
            print(f"step {step:3d}  train env-CE {train_ce:.3f}")

    print("\n" + "=" * 64)
    print("RESULT - Tinker trained the model to predict the environment")
    print("=" * 64)
    print(f"held-out env-token CE: {before:.3f} -> {after:.3f} nats/token")

    if args.checkpoint_name:
        checkpoint = training_client.save_state(args.checkpoint_name).result()
        print(f"saved checkpoint: {checkpoint.path}")


if __name__ == "__main__":
    main()
