# Pathway Analysis Environment

`pathway_analysis_env` is an OpenEnv environment for evaluating multi-step,
tool-using agents on a realistic scientific workflow.

In plain terms: each task gives an agent a gene-expression dataset with labeled
sample groups (for example treatment vs control). The agent must:

1. inspect and understand the experiment design,
2. run **differential expression** (find genes that changed between groups),
3. run **pathway enrichment** (map changed genes to higher-level biological
   programs),
4. submit a final pathway hypothesis.

This environment is designed to test agent behavior that matters in production:
sequencing decisions correctly, using tools in the right order, handling
structured outputs, and producing evidence-backed conclusions rather than
one-shot guesses.

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

## Add tasks from public databases (GEO)

You can add more real-world tasks from public studies (for example NCBI GEO) by
creating one case folder per study/contrast and then registering it in a manifest.

Recommended process:

1. **Pick a study + contrast**
   - Choose clear conditions (for example treated vs control).
   - Prefer datasets with enough replicates per group.

2. **Prepare a count matrix**
   - Save gene-by-sample counts as `.csv.gz`.
   - Put it under `envs/pathway_analysis_env/data/geo_eval/<task_id>/`.

3. **Create case JSON**
   - Add `<task_id>_case.json` in the same folder.
   - Include at minimum:
     - `experiment_metadata` (`accession`, `reference`, `summary`)
     - `counts_file`
     - `sample_ids`
     - `sample_metadata` (sample -> condition)
     - `conditions`
     - `default_contrast`
     - `enrichr_libraries`
   - You can use existing GEO cases as templates:
     - `data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json`
     - `data/geo_eval/gse111151_tamoxifen_benchmark/gse111151_case.json`
     - `data/geo_eval/gse216540_tpm_pseudo_benchmark/gse216540_case.json`

4. **Add the task to an eval manifest**
   - Update or create a manifest under `data/` (for example `data/eval_manifest_geo3.json`).
   - Add an episode pointing to your new case file.

5. **Run and validate**
   - Run environment eval:
     - `PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_agent_eval_suite.py --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json`
   - Run LLM eval and (optionally) judge:
     - `PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json`
     - `PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_judge.py --case geo_eval/<task_id>/<task_case.json> --agent-model gpt-5 --judge-model gpt-5`

Notes:

- Keep large intermediate artifacts (`de_all.json`, `enrichment.json`, `work/`) out of commits unless needed.
- For reproducible benchmark tasks, commit the case JSON and minimal count input needed to rerun.

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
