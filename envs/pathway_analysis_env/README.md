# Pathway Analysis Environment

A toy pathway analysis environment for [OpenEnv](https://github.com/meta-pytorch/OpenEnv) that models a simple computational biology task: identifying the activated signaling pathway from synthetic omics data.

## Overview

Each episode hides a **true activated pathway** (e.g. "MAPK signaling"). The agent can inspect the dataset, run differential expression analysis, run pathway enrichment, and then submit a hypothesis. Rewards are deterministic and designed for validating environment mechanics, not biological realism.

## Actions

| Action | Description | Reward |
|--------|-------------|--------|
| `inspect_dataset` | View dataset metadata and available conditions | +0.1 |
| `run_differential_expression` | Run DE analysis to get top differentially expressed genes | +0.3 |
| `run_pathway_enrichment` | Run enrichment to get top enriched pathways | +0.5 |
| `submit_answer` | Submit a pathway hypothesis (correct: +2.0, incorrect: -1.0) | +2.0 / -1.0 |

## Quick Start

### Install

```bash
cd envs/pathway_analysis_env
pip install -e .
```

### Direct test (no server)

```bash
PYTHONPATH=src:envs python -c "
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment
from pathway_analysis_env.models import PathwayAction

env = PathwayEnvironment()
obs = env.reset()
print(obs.message)

obs = env.step(PathwayAction(action_type='inspect_dataset'))
print(obs.message, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='run_differential_expression'))
print(obs.message, obs.top_genes, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='run_pathway_enrichment'))
print(obs.message, obs.top_pathways, 'reward:', obs.reward)

obs = env.step(PathwayAction(action_type='submit_answer', hypothesis='MAPK signaling'))
print(obs.message, 'done:', obs.done, 'reward:', obs.reward)
"
```

### Run the HTTP server

```bash
uvicorn pathway_analysis_env.server.app:app --host 0.0.0.0 --port 8000
```

### Connect via client

```python
import asyncio
from pathway_analysis_env import PathwayEnv, PathwayAction

async def main():
    async with PathwayEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(result.observation.message)

        result = await client.step(PathwayAction(action_type="inspect_dataset"))
        result = await client.step(PathwayAction(action_type="run_differential_expression"))
        result = await client.step(PathwayAction(action_type="run_pathway_enrichment"))
        result = await client.step(PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling"))
        print("done:", result.done, "reward:", result.reward)

asyncio.run(main())
```

## Roadmap

| Version | Change | Why |
|---------|--------|-----|
| v2 | Replace hard-coded outputs with precomputed CSV/JSON from a real expression dataset | Keeps behavior deterministic while feeling more realistic |
| v3 | Swap analysis actions to call real DE and enrichment tools (PyDESeq2, GSEApy) | Turns the toy env into a real computational workflow env |
| v4 | Add multi-omics tasks, richer rewards, and BixBench-inspired task templates | Moves closer to useful scientific-agent evaluation |
