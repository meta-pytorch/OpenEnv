# Harness rollout worker for online RL with TRL (draft)

A runnable draft example of training an OpenEnv-style **agentic harness** (an agent that owns its own
loop) with TRL's online RL (AsyncGRPO). It runs today with **no GPU** (fake mode) and with **real
vLLM**.

This example ships in the same PR as the minimal contract it builds on (`src/openenv/core/harness/`:
`rollout.py` + `interception.py`), so the whole dynamic is reviewable in one place. The contract is the
OpenEnv contribution. The **worker is the trainer side**: for a real integration it gets **vendored
into TRL** (the worker conforms to TRL's `RolloutWorkerProtocol`, so it belongs with the trainer). It
is shown here so the contract is not just interfaces but a thing you can read and run.

The example vendors `rollout.py` + `interception.py` locally so it stays self-contained (runs with just
`aiohttp` + `requests`, no heavy imports) and is copy-pasteable into TRL as-is. Those two files mirror
the canonical contract in `src/openenv/core/harness/`. The richer interception/sandbox stack lives in
OpenEnv PR #694.

To pick it up: implement the two seams on the trainer side and wire it into AsyncGRPO. See
`trl_adapter.py` for the exact spots.

## The idea

A harness is an agent that owns its loop (it calls the model, uses tools, decides when to stop). To
train it on-policy you must run its LLM calls through the policy being trained and capture the exact
tokens. This worker drives the harness through an interception proxy, scores the episode with the
env's `verify()`, and emits a **message-level rollout**. It does NOT tokenize. Tokenization (the
prefix-preserving, token-in/token-out work) is the trainer's job, behind the `generate` seam.

```
harness (owns its loop) --HTTP--> interception proxy --> HarnessRolloutWorker
   ^                                                       |- generate(...)   [Seam 1: trainer]
   |---------------- completion ---------------------------|- verify() -> reward
                                                           |- emit RolloutMessages [Seam 2: trainer consumes]
```

## The two seams (what the trainer implements)

- **Seam 1:** `generate(rollout_id, turn, messages, tools, sampling) -> completion_text`. The trainer
  generates with its inference engine and records token_ids + logprobs keyed by `(rollout_id, turn)`.
  This is also where messages-to-tokens / TITO lives.
- **Seam 2:** the worker emits `RolloutMessages{rollout_id, messages, reward}`. The trainer stitches
  the captured per-turn tokens into a training sample, applies the reward, computes the advantage.

## What is done vs to do

**Done (this example):**
- `HarnessRolloutWorker`: drives the harness via interception, `verify()`, emits message-level rollouts.
- A minimal interception proxy (the richer one is OpenEnv PR #694).
- A harness with the right dynamic (a ReAct agent that owns its loop, multi-turn, interception-gated).
- Runnable with no GPU (fake) and with real vLLM (real token_ids + logprobs capture).

**To do (trainer side):** see `trl_adapter.py`. Implement `generate` (Seam 1, with TITO) and wrap the
worker into TRL's `RolloutWorkerProtocol` (Seam 2 + wiring into `AsyncGRPOTrainer(rollout_worker=...)`).

> Weight sync is a trainer-side concern. TRL handles it on the default path (`environment_factory` +
> the built-in `AsyncRolloutWorker`). When you inject a custom `rollout_worker`, TRL sets
> `weight_transfer = None` and `_sync_weight()` becomes a no-op, so the trainer side must wire weight
> sync itself (worker-driven, the way OpenEnv PR #695 does it with `send_weights` +
> `update_model_version`, or by extending `_sync_weight`). It does not touch OpenEnv. See
> `trl_adapter.py`. This draft validates the rollout dynamic and the seam, not a converged training run.

## Run

```bash
pip install aiohttp requests           # fake mode deps
# transformers + vllm are only needed for vllm mode

# no GPU: scripted generator driving the real harness dynamic
python run.py --mode fake

# GPU box, with `vllm serve <model> --port 8000` already running:
python run.py --mode vllm --vllm-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct --dump captures.json

# tests (no GPU)
pytest test_rollout_worker.py
```

`--dump captures.json` (vllm mode) writes the real per-turn `(prompt_ids, completion_ids, logprobs)` to
JSON: the exact data a TITO step would stitch into a training sample.

## Files

Vendored minimal contract + transport (mirror the canonical contract in `src/openenv/core/harness/`):

| File | Role |
|------|------|
| `rollout.py` | The contract: `AgentSession`, `AgentSessionFactory`, `RolloutMessages`, `GenerateAPI`. |
| `interception.py` | `InterceptionServer`: the OpenAI-compatible gating proxy. |

The worker + the example:

| File | Role |
|------|------|
| `rollout_worker.py` | `HarnessRolloutWorker` (the trainer side): drives the harness, emits message-level rollouts. |
| `harness.py` | The harness (ReAct agent owning its loop) + session + arithmetic task + verifier. |
| `generate.py` | `FakeGenerate` (no GPU) and `VLLMGenerate` (real vLLM + token capture), both Seam 1. |
| `trl_adapter.py` | Skeleton of the trainer-side integration (where Seam 1/2 + AsyncGRPO wiring go). |
| `run.py` | Entry point, `--mode fake|vllm`. |
| `test_rollout_worker.py` | Unit tests, no GPU. |

## Notes

- `rollout.py` (contract) and `interception.py` are vendored here so the example is self-contained and
  directly portable into TRL. They mirror the canonical contract in `src/openenv/core/harness/`.
- The worker is harness-agnostic: it does not change when you swap the harness. The ReAct agent here
  gives the right dynamic without a full coding agent. A follow-up can swap it for a real harness
  (Pi / OpenCode) over OpenEnv's interception + sandbox (PR #694).
- The core (worker + interception + message-level rollout) is framework-neutral. TRL is the first
  integration, not the only one.
