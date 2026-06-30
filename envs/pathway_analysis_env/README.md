# Pathway Analysis Environment

`pathway_analysis_env` is an OpenEnv environment for evaluating tool-using agents
on a realistic RNA-seq-style analysis loop.

Each task gives:
- a gene-expression matrix (`counts_file`)
- sample groups (`sample_metadata`)
- a default contrast (reference vs alternate condition)

The agent must execute:
1. inspect/understand design
2. differential expression (which genes changed)
3. pathway enrichment (which biological programs are implicated)
4. submit a final pathway hypothesis

## Quick start

From repo root:

```bash
uv sync --all-extras
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_agent_eval_suite.py \
  --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json
```

Run LLM eval:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py \
  --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json
```

Run LLM judge for one case:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_judge.py \
  --case geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json \
  --agent-model gpt-5 \
  --judge-model gpt-5
```

## Add a new GEO task (2 commands)

### 1) Create case + copy counts

Prepare a metadata CSV with columns:
- `sample_id`
- `condition`

Then run:

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/create_geo_task.py \
  --task-id gseXXXX_example \
  --accession GSEXXXX \
  --summary "One-line study summary" \
  --counts-file /absolute/path/to/counts.csv.gz \
  --metadata-csv /absolute/path/to/samples.csv \
  --reference-condition control \
  --alternate-condition treated
```

This creates:
- `data/geo_eval/gseXXXX_example/gsexxxx_case.json`
- `data/geo_eval/gseXXXX_example/<counts file>`

### 2) Add it to a manifest

```bash
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/append_task_to_manifest.py \
  --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json \
  --episode-id geo_gseXXXX_example \
  --case-file geo_eval/gseXXXX_example/gsexxxx_case.json \
  --hypothesis "expected biological theme"
```

Now rerun eval on that manifest.

## Scripts

- `scripts/create_geo_task.py` — create a GEO case from counts + metadata.
- `scripts/append_task_to_manifest.py` — append/update one episode in a manifest.
- `scripts/run_agent_eval_suite.py` — run scripted environment evaluation.
- `scripts/run_llm_agent_eval.py` — run tool-calling LLM evaluation.
- `scripts/run_llm_judge.py` — score report quality with an LLM judge.
- `scripts/export_agent_safe_cases.py` — export secret-stripped case files.

## Notes

- Keep large intermediate artifacts (`de_all.json`, `enrichment.json`, `work/`) out of commits unless required.
- For reproducibility, commit only case JSON + minimal raw inputs (counts/metadata mapping) needed to rerun.
- Eval defaults are documented in `docs/AGENT_EVAL.md`.
