# Agent evaluation guide — pathway_analysis_env

This guide covers how to evaluate tool-calling LLM agents in `pathway_analysis_env`.

## Eval defaults

`reset()` enables `eval_mode=True` by default.

In eval mode:

- `true_pathway` is hidden from agent-visible state
- `submit_answer` requires DE + ORA first
- custom ORA `gene_list` injection is blocked
- intermediate shaping rewards are flattened
- step budget is enforced (`max_steps`, default 30)

For local debugging only, set `eval_mode=False`.

## Standard environment eval

Run the manifest-driven harness:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_agent_eval_suite.py
```

Default manifest: `envs/pathway_analysis_env/data/eval_manifest.json`.

## LLM eval (tool-calling)

Set one provider credential in env or `.env`:

- `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`

Then run:

```bash
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py
```

Useful flags:

```bash
# Choose provider explicitly
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py --provider groq

# Override models
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py --models gpt-5

# Use custom manifest
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json
```

Reports are written to:

- `envs/pathway_analysis_env/outputs/llm_eval/latest.json`
- `envs/pathway_analysis_env/outputs/llm_eval/latest.md`

## LLM judge eval (report quality)

Run agent + judge against a single case:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_judge.py \
  --case geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json \
  --agent-model gpt-5 \
  --judge-model gpt-5 \
  --out-json envs/pathway_analysis_env/outputs/llm_eval/live_judge_gse128911_gpt5_gpt5.json
```

## Scoring access (orchestrator side)

After submit:

```python
outcome = env.episode_outcome
```

With HTTP server:

- `GET /orchestrator/episode_outcome`
- `GET /orchestrator/eval_protocol`

## Agent-safe case export

Export sanitized cases for agent-facing deployments:

```bash
cd envs/pathway_analysis_env
PYTHONPATH=../../src:.. uv run python scripts/export_agent_safe_cases.py
```

Output path: `data/agent_safe/`.

## Failure codes

Failure code definitions are in `docs/FAILURE_CODES.md`.
