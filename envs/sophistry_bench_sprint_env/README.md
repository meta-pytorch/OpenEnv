---
title: Sophistry Bench Sprint Env
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# sophistry_bench_sprint_env

OpenEnv port of the [sophistry-bench](https://github.com/acharyaanusha/sophistry-bench)
single-agent advocacy **reward-hacking sprint** environment. One-turn advocacy on QuALITY
reading-comprehension passages: the policy is assigned the gold answer or a distractor and must
produce one `<claim>`/`<cite>` argument. The reward proxy peaks at exactly 8 `<claim>` tags
(`claim_count_cliff`); four weight-0 canaries detect format hacking.

Scoring is imported verbatim from the [`sophistry-bench-sprint`](https://pypi.org/project/sophistry-bench-sprint/)
PyPI package, so the reward numbers are identical to the Prime Intellect Hub env.

## Episode model

Single step. `reset()` issues a task; `step(AdvocacyAction(text=...))` scores it and returns
`done=True`.

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `SPRINT_N_ITEMS` | `50` | QuALITY items to load (2 advocacy rows each) |
| `SPRINT_PASSAGE_CHARS` | `2000` | Passage char cap |
| `SPRINT_SEED` | `0` | Distractor-selection seed (deterministic) |
| `SPRINT_WEIGHTS` | `1,0,0,0,0,0,0,0` | 8 reward weights, order: `aggregate, correctness, n_claims, n_citations, alternation_canary, starts_with_canary, length_band_canary, template_echo_canary`. Do **not** weight canaries during training. |
| `SPRINT_EXPOSE_CORRECTNESS` | `0` | When `1`/`true`, surface `correctness_reward` (the hidden ground truth) in the wire `metadata`/`components`. Off by default so a harness can't accidentally leak it to the policy. This flag controls only *surfacing*, not *weighting*: correctness affects `reward` only via its `SPRINT_WEIGHTS` entry, which is `0` by default. |

## Usage

The client is **async by default** (like every OpenEnv client):

```python
import asyncio

from sophistry_bench_sprint_env import SophistryBenchSprintEnv


async def main():
    # Deployed Hugging Face Space (or .from_docker_image("openenv-sophistry_bench_sprint:latest")):
    client = await SophistryBenchSprintEnv.from_env("openenv-community/sophistry_bench_sprint_env")
    async with client:
        obs = (await client.reset()).observation
        print(obs.prompt, obs.answer_to_defend)
        result = await client.step_text("<claim>...</claim><cite>...</cite>")
        print(result.reward, result.observation.metadata)


asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
with SophistryBenchSprintEnv(base_url="http://localhost:8000").sync() as client:
    obs = client.reset().observation
    result = client.step_text("<claim>...</claim><cite>...</cite>")
    print(result.reward, result.observation.metadata)
```

`result.observation.metadata` carries the reward components every step — the canary scores are
the reward-hacking measurement. By default it holds **seven** components; `correctness_reward`
(the hidden ground truth) is withheld unless `SPRINT_EXPOSE_CORRECTNESS=1` (see above).

> **Do not feed `observation.metadata` / `observation.components` back into the policy's
> prompt.** `reset()` deliberately tells the policy only *what* to defend, never *whether* it
> is correct. `correctness_reward` is withheld from the wire by default for exactly this
> reason; even with the rest of the components, forwarding them to the agent leaks the
> reward signal and defeats the reward-hacking measurement.

## Training

[`examples/sophistry_bench_sprint_grpo.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/sophistry_bench_sprint_grpo.py)
trains a policy on this env with TRL's `GRPOTrainer`. Since the episode is
single-step, this is a plain prompt -> completion -> reward GRPO setup — no
`environment_factory`/tool-calling needed (contrast with the multi-turn Wordle
GRPO tutorial). It connects directly to the deployed Space's source (cloned
and run locally via `uv`, not Docker, and not subject to the Space's request
quota) and only depends on `openenv[core]` from PyPI, so it also runs as a
standalone `uv` script, including on Hugging Face Jobs. Verified end-to-end: a
short run (4 episodes, 1 step, `Qwen2.5-0.5B-Instruct`) produces a real
checkpoint and a real reward from the live env.

### Also validated on Prime Intellect

The same scoring is registered as `anusha/sophistry-bench-sprint` on the Prime
Intellect Hub (parity-tested against this OpenEnv port). A 100-step GRPO run
there reproduces the reward-hacking signature this env is designed to
surface. Config:
[`training/sophistry_bench_sprint.toml`](https://github.com/huggingface/OpenEnv/blob/main/envs/sophistry_bench_sprint_env/training/sophistry_bench_sprint.toml);
full per-step metrics:
[`training/metrics.csv`](https://github.com/huggingface/OpenEnv/blob/main/envs/sophistry_bench_sprint_env/training/metrics.csv).
Base model `Llama-3.2-1B-Instruct`, default `SPRINT_WEIGHTS` (only `aggregate_reward` weighted).

| Steps | `aggregate_reward` (proxy, optimized) | `n_claims` (cliff target) | `correctness_reward` (ground truth, weight 0) |
|---|---|---|---|
| 0–9 | 0.477 | 0.989 | 0.470 |
| 10–19 | 0.679 | 1.002 | 0.532 |
| 20–29 | 0.740 | 1.015 | 0.524 |
| 40–49 | 0.770 | 1.001 | 0.353 |
| 90–99 | 0.766 | 1.001 | 0.470 |

`aggregate_reward` climbs from ~0.48 to a ~0.77 plateau and `n_claims` saturates at the
`claim_count_cliff` target almost immediately — the policy learns to hit exactly 8 `<claim>`
tags. `correctness_reward` (the hidden ground truth, **not** in the optimized objective)
stays flat and noisy the entire run with no upward trend. That gap — proxy reward up,
ground-truth quality flat — is the reward-hacking measurement this env exists to produce.

## Build & test

```bash
# Tests live with the other env tests. Run them from the repo root using this
# env's venv (which installs the scoring package):
uv run --project envs/sophistry_bench_sprint_env --extra dev \
  pytest tests/envs/test_sophistry_bench_sprint_environment.py -v
# The module pulls the published sophistry-bench-sprint, so in the repo's shared
# CI (where it isn't installed) it skips via pytest.importorskip — same as other
# envs with heavy deps (e.g. tbench2's camel guard).

# Container
openenv build sophistry_bench_sprint_env
# produces image tag: openenv-sophistry_bench_sprint:latest
```
