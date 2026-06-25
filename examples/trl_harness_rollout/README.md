# Harness rollout worker for online RL with TRL (draft)

A draft to enable integrating OpenEnv **agentic harnesses** (agents that own their own loop) with
TRL's online RL (AsyncGRPO). It provides the **OpenEnv side** end to end and a clear, narrow **seam**
where the TRL side plugs in. It runs today with **no GPU** (fake mode) and with **real vLLM**.

This is a draft to make the integration easy to pick up: download it, implement the two seams on the
trainer side, and wire it into AsyncGRPO. See `trl_adapter.py` for the exact spots to implement.

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

**Done (OpenEnv side, this draft):**
- `HarnessRolloutWorker`: drives the harness via interception, `verify()`, emits message-level rollouts.
- An interception proxy (a clean stand-in for OpenEnv's `InterceptionServer`, PR #694).
- A harness with the right dynamic (a ReAct agent that owns its loop, multi-turn, interception-gated).
- Runnable with no GPU (fake) and with real vLLM (real token_ids + logprobs capture).

**To do (trainer side):** see `trl_adapter.py`. Implement `generate` (Seam 1, with TITO) and wrap the
worker into TRL's `RolloutWorkerProtocol` (Seam 2 + wiring into `AsyncGRPOTrainer(rollout_worker=...)`).

> Not full on-policy training yet. In TRL today, injecting a custom `rollout_worker` sets
> `weight_transfer = None` and `_sync_weight()` becomes a no-op (it does not call
> `rollout_worker.send_weights`). Until the trainer side handles weight sync (worker-driven, or by
> extending `_sync_weight`), generation can drift off-policy. See `trl_adapter.py`. This draft
> validates the rollout dynamic and the seam, not a converged training run.

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

In OpenEnv core (reusable, this PR adds them under `src/openenv/core/harness/`):

| File | Role |
|------|------|
| `rollout_worker.py` | `HarnessRolloutWorker` + the contract (`GenerateAPI`, `AgentSession`, `RolloutMessages`). |
| `interception.py` | `InterceptionServer`: the OpenAI-compatible gating proxy. |

In this example (`examples/trl_harness_rollout/`):

| File | Role |
|------|------|
| `harness.py` | The harness (ReAct agent owning its loop) + session + arithmetic task + verifier. Imports from core. |
| `generate.py` | `FakeGenerate` (no GPU) and `VLLMGenerate` (real vLLM + token capture), both Seam 1. |
| `trl_adapter.py` | Skeleton of the trainer-side integration (where Seam 1/2 + AsyncGRPO wiring go). |
| `run.py` | Entry point, `--mode fake|vllm`. |
| `test_rollout_worker.py` | Unit tests, no GPU. |

## Notes

- The worker and the interception live in `openenv.core.harness` (reusable by any env and any trainer
  adapter). The example imports them. The harness-agnostic worker does not change when you swap the
  harness.
- The ReAct agent here gives the right dynamic without a full coding agent. A follow-up can swap it for
  a real harness (Pi / OpenCode). The worker does not change.
- A richer interception/sandbox stack is proposed in PR #694, and PR #695 has an environment-specific
  worker. This adds a clean, generic core worker + interception, to be reconciled with #694.
- The core (worker + interception + message-level rollout) is framework-neutral. TRL is the first
  integration, not the only possible one.
