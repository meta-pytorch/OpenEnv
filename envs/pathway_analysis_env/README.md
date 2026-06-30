# Pathway Analysis Environment

`pathway_analysis_env` is an OpenEnv environment for pathway-analysis reasoning.
An agent gets expression data, runs differential expression and pathway enrichment,
then submits a pathway hypothesis.

## What the agent can do

The environment exposes six actions:

- `inspect_dataset` — inspect samples, conditions, and availability checks.
- `understand_experiment_design` — summarize groups and optionally validate a contrast.
- `run_differential_expression` — run PyDESeq2 for a chosen reference vs alternate group.
- `run_pathway_enrichment` — run ORA against case pathway gene sets.
- `compare_pathways` — compare overlap support between two pathways.
- `submit_answer` — submit the final pathway hypothesis.

Typical workflow:

1. Inspect design
2. Validate or choose contrast
3. Run differential expression
4. Run enrichment
5. Submit answer

## Data modes

- **Pipeline cases**: include counts + sample metadata and run real DE/ORA.
- **Legacy cases**: use fixed outputs for lightweight tests.
- **GEO cases**: real-study-style tasks with report-oriented evaluation.

## Evaluation behavior (default)

`reset()` runs with `eval_mode=True`, which enforces benchmark-safe behavior:

- Hidden answer (no `true_pathway` in agent-visible state)
- Required workflow (`submit_answer` requires DE + ORA)
- No ORA gene-list injection in eval mode
- No reward shaping (intermediate rewards are zeroed)
- Step budget via `max_steps`

For debugging or demos, set `eval_mode=False`.

For details, see:

- `docs/AGENT_EVAL.md`
- `docs/FAILURE_CODES.md`

## Quick start

### 1) Install

From repo root:

```bash
uv sync --all-extras
```

### 2) Run server + UI

```bash
cd envs/pathway_analysis_env
uv run server
```

Open `http://localhost:8000/web/` and use the **Visualization** tab.
UI walkthrough: `docs/USER_GUIDE_UI.md`.

### 3) Run a local eval

From repo root:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_agent_eval_suite.py
```

### 4) Run LLM agent eval

Set one provider key (`GROQ_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`),
then run:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py
```

Outputs are written to `envs/pathway_analysis_env/outputs/llm_eval/`.

## Important scripts

- `scripts/run_agent_eval_suite.py` — manifest-driven environment eval.
- `scripts/run_llm_agent_eval.py` — tool-calling LLM evaluation.
- `scripts/generate_curriculum_cases.py` — generate curriculum tasks.
- `scripts/export_agent_safe_cases.py` — strip secrets and export safe cases.
- `bench/geo_agent_eval.py` — GEO-oriented evaluation utility.
- `scripts/run_llm_judge.py` — report-vs-reference LLM judging.

## Minimal client example

```python
import asyncio
from pathway_analysis_env import PathwayEnv, PathwayAction

async def main():
    async with PathwayEnv(base_url="http://localhost:8000") as client:
        await client.reset()
        await client.step(PathwayAction(action_type="understand_experiment_design"))
        await client.step(PathwayAction(action_type="run_differential_expression"))
        await client.step(PathwayAction(action_type="run_pathway_enrichment"))
        result = await client.step(
            PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling")
        )
        print(result.done, result.reward)

asyncio.run(main())
```
