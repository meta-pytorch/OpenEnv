# LLM Judge Evaluation (Pure Report Comparison)

This document describes the non-training evaluation workflow for comparing an
agent's written analysis report against a reference report.

## What This Is

- **Purpose:** richer scientific assessment than keyword matching.
- **Scope:** eval-only (offline analysis), not environment reward shaping.
- **Judge input:** agent report + reference report.
- **Judge output:** 0-1 scores for:
  - primary_biology
  - supporting_pathways
  - evidence_grounding
  - mechanism
  - overall

## Important Design Choice

The reference report is generated from the **same live episode outputs** (DE and
ORA) that the agent saw during that run. This avoids mismatches between:

- stale precomputed `enrichment.json` artifacts, and
- live Enrichr results at evaluation time.

## Scripts

- `scripts/run_llm_judge.py`
  - Runs one case with a tool-calling agent model.
  - Builds live reference from that same episode.
  - Calls a judge model and writes JSON artifact.
- `scripts/build_judge_pdf.py`
  - Builds a visual PDF from a judge artifact JSON.

## Example Commands

Run a single case:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:envs uv run python \
  envs/pathway_analysis_env/scripts/run_llm_judge.py \
  --case geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json \
  --agent-model gpt-5 \
  --judge-model gpt-4o \
  --out-json envs/pathway_analysis_env/outputs/llm_eval/live_judge_gse128911_gpt5.json
```

Build a PDF:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src:envs uv run python \
  envs/pathway_analysis_env/scripts/build_judge_pdf.py \
  --artifact envs/pathway_analysis_env/outputs/llm_eval/live_judge_gse128911_gpt5.json \
  --enrichment envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/enrichment.json \
  --out envs/pathway_analysis_env/outputs/llm_eval/live_judge_gse128911_gpt5.pdf
```

## Notes

- LLM judging is **non-deterministic** and should not replace deterministic
  environment rewards for RL training.
- Keep judge model separate from agent model when possible (reduces self-bias).
- Do not commit secrets; keep API keys in `.env` (gitignored).
