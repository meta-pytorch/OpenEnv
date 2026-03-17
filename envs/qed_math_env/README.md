---
title: QED Math Environment
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - mathematics
  - proof-evaluation
  - llm-grading
---

# QED Math Environment

Mathematical proof generation and evaluation environment for OpenEnv, ported from [QED-Nano](https://github.com/meta-pytorch/QED-Nano). Agents receive math problems, submit proofs, and receive LLM-based rubric grading on a 0–7 scale with normalized rewards.

## Features

- **LLM-based rubric grading** (0–7 scale) via any OpenAI-compatible endpoint
- **Answer-mode verification** using `math_verify` for `\boxed{}` answers
- **Flexible dataset loading**: local JSONL/JSON, Hugging Face Hub, or built-in bootstrap problems
- **Reward shaping**: discount factor, length penalty, and optional score thresholding
- **Reasoning stripping**: configurable delimiters (e.g. `<think>...</think>`) removed before grading
- **Multi-step problems**: configurable max attempts with per-attempt feedback
- **Verifier metrics**: QED-Nano-compatible metrics surfaced in observation metadata, ready for TrackIO / WandB
- **MCP tool interface**: `get_problem`, `submit_proof`, `get_grading_guidelines`

## Quick Start

### Async (default)

```python
import asyncio
from qed_math_env import QEDMathEnv

async def main():
    async with QEDMathEnv(base_url="http://localhost:8000") as env:
        # Reset to load a problem
        result = await env.reset()
        obs = result.observation
        print(f"Problem: {obs.problem[:100]}...")

        # Submit a proof
        submission = await env.submit_proof(proof="By induction on n...")
        print(f"Score: {submission.score}/7, Reward: {submission.reward:.2f}")

asyncio.run(main())
```

### Sync

```python
from qed_math_env import QEDMathEnv

with QEDMathEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    submission = env.call_tool("submit_proof", proof="By induction on n...")
```

### MCP tool-calling

```python
async with QEDMathEnv(base_url="http://localhost:8000") as env:
    await env.reset()

    # Discover tools
    tools = await env.list_tools()
    print([t.name for t in tools])
    # ['get_problem', 'submit_proof', 'get_grading_guidelines']

    # Call tools by name
    problem = await env.call_tool("get_problem")
    guidelines = await env.call_tool("get_grading_guidelines")
    result = await env.call_tool("submit_proof", proof="...", output_length_tokens=256)
```

## Building & Running

```bash
# Build Docker image (from project root)
docker build -t qed-math-env:latest -f envs/qed_math_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY qed-math-env:latest

# Or run locally with uvicorn
PYTHONPATH=src:envs uvicorn qed_math_env.server.app:app --port 8000

# Or install and run via uv
cd envs/qed_math_env
uv sync
uv run server
```

## Project Structure

```
qed_math_env/
├── __init__.py              # Module exports (QEDMathEnv, models)
├── models.py                # ProblemObservation, ProofSubmissionObservation
├── client.py                # QEDMathEnv client (MCPToolClient subclass)
├── openenv.yaml             # OpenEnv manifest with metrics declarations
├── pyproject.toml           # Dependencies
├── uv.lock                  # Locked dependencies
├── README.md
├── prompts/
│   └── evaluator_prompts/
│       └── v2.md            # Evaluator prompt template (QED-Nano v2)
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI server (create_app factory)
    ├── qed_math_environment.py  # QEDMathEnvironment (MCPEnvironment)
    ├── mcp_server.py        # MCP tool registration
    ├── rubric.py            # MathProofRubric + GradingResult
    └── Dockerfile
```

## Configuration

The environment is configured via `QEDMathConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_path` | `None` | Dataset source: local path, Hub ID, or list of specs. `None` uses bootstrap problems. |
| `grader_model` | `"gemini-3-pro"` | Model identifier for the LLM grader (any OpenAI-compatible endpoint) |
| `prompt_name` | `"v2"` | Evaluator prompt template name (loads from `prompts/evaluator_prompts/`) |
| `custom_reward_threshold` | `False` | When `True`, collapses partial-credit scores 1–5 → 1 |
| `max_attempts` | `1` | Max proof attempts per problem (>1 for multi-step) |
| `discount_factor` | `1.0` | Exponential discount: `reward *= discount_factor ** output_length_tokens` |
| `buffer_tokens` | `0` | Length penalty zone width. `0` disables the penalty. |
| `max_tokens` | `0` | Max token limit for length penalty computation |
| `reasoning_delimiters` | `None` | Delimiter strings to strip reasoning (e.g. `["</think>"]`) |

Environment variables:
- `OPENAI_API_KEY` — API key for the grader LLM
- `OPENAI_BASE_URL` — Base URL override (for non-OpenAI providers)

## Dataset Format

### Local JSONL/JSON

```json
{
  "problem": "Prove that the sum of two even integers is even.",
  "solution": "Let a=2m and b=2n. Then a+b=2(m+n), which is even.",
  "rubrics": [
    {"title": "Definitions", "points": 2, "desc": "Correctly defines even integers."},
    {"title": "Algebra", "points": 3, "desc": "Valid algebraic manipulation."},
    {"title": "Conclusion", "points": 2, "desc": "Correctly concludes evenness."}
  ],
  "dataset": "FineProofs-RL",
  "problem_id": "fp_001"
}
```

### Hugging Face Hub

```python
QEDMathConfig(dataset_path="meta-math/MetaMathQA")
# or with config
QEDMathConfig(dataset_path={"hub_id": "meta-math/MetaMathQA", "split": "train", "config": "default"})
```

### Field Aliases

The environment normalizes many dataset formats automatically:

| Canonical Field | Accepted Aliases |
|----------------|------------------|
| `problem` | `question`, `problem_statement`, `prompt` |
| `reference_solution` | `solution`, `human_solution`, `answer`, `ground_truth` |
| `grading_guidelines` | `rubrics`, `marking_scheme`, `rubric` |
| `problem_id` | `id`, `uid`, `index` |
| `original_problem` | Used for RC-stream problems where the actor prompt differs from grading prompt |

## Observation Space

### ProblemObservation (from `reset` / `get_problem`)

| Field | Type | Description |
|-------|------|-------------|
| `problem` | `str` | Math problem statement |
| `reference_solution` | `str` | Ground-truth solution |
| `grading_guidelines` | `str` | Rubric / marking scheme |
| `problem_id` | `str` | Unique identifier |
| `problem_type` | `str` | `"proof"`, `"answer"`, or `"multi_step"` |
| `dataset_source` | `str` | Source dataset name |
| `metadata` | `dict` | Additional context (e.g. `original_problem`) |

### ProofSubmissionObservation (from `submit_proof`)

| Field | Type | Description |
|-------|------|-------------|
| `proof` | `str` | Submitted proof text |
| `score` | `int` | Raw grade (0–7) |
| `feedback` | `str` | Full grader response |
| `reward` | `float` | Shaped reward in [0, 1] |
| `done` | `bool` | Whether the episode is over |
| `is_correct` | `bool` | Whether score >= success threshold (default 6) |
| `attempt_number` | `int` | Current attempt count |
| `attempts_remaining` | `int` | Remaining attempts |
| `problem_type` | `str` | Problem type |
| `metadata` | `dict` | Contains `verifier_metrics`, `base_reward`, `shaped_reward` |

## MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_problem` | Return current problem statement and metadata | — |
| `submit_proof` | Submit a proof for LLM-based rubric grading | `proof` (str, required), `output_length_tokens` (int, optional) |
| `get_grading_guidelines` | Return the rubric/marking scheme | — |

## Reward Computation

The reward pipeline follows QED-Nano conventions:

1. **LLM grading**: Score 0–7 via evaluator prompt with `<score>N</score>` parsing
2. **Optional thresholding**: Collapses 1–5 → 1 (when `custom_reward_threshold=True`)
3. **Normalization**: `reward = score / 7.0`
4. **Discount factor**: `reward *= discount_factor ** output_length_tokens`
5. **Length penalty**: Linear penalty when output approaches `max_tokens`

For answer-mode problems (`evaluation_mode: "answer"`), `math_verify` is used instead of LLM grading: `\boxed{}` answers are extracted and verified against the gold answer.

## Verifier Metrics

Every `submit_proof` call emits verifier metrics in `metadata["verifier_metrics"]`, compatible with TrackIO and WandB:

| Metric | Description |
|--------|-------------|
| `verifier/rollouts/success` | 1 if grading succeeded |
| `verifier/rollouts/failure` | 1 if grading failed |
| `verifier/failures/timeout` | Count of timeout errors |
| `verifier/failures/rate_limit` | Count of rate-limit errors |
| `verifier/failures/no_input` | 1 if proof was empty |
| `verifier/failures/no_score_tag` | 1 if LLM response had no `<score>` tag |
| `verifier/failures/all_attempts_failed` | 1 if all retries exhausted |
| `verifier/failures/num_retries` | Number of retries used |
| `verifier/runtime/latency_per_request` | Grading wall-clock time (seconds) |
| `verifier/runtime/input_tokens` | Estimated input tokens |
| `verifier/runtime/output_tokens` | Estimated output tokens |
| `reward/base` | Pre-shaping reward |
| `reward/shaped` | Post-shaping reward |
| `reward/score_raw` | Raw integer score (0–7) |
| `reward/overlong_penalty` | Length penalty applied |
| `episode/attempt_number` | Current attempt |
| `episode/is_correct` | 1 if correct |
| `episode/problem_type` | proof / answer / multi_step |
| `episode/dataset_source` | Source dataset name |

### TrackIO Integration

```python
import trackio

run = trackio.init(project="qed-math-training")

# After each submit_proof call:
verifier_metrics = result["metadata"]["verifier_metrics"]
numeric = {k: v for k, v in verifier_metrics.items() if isinstance(v, (int, float))}
run.log(numeric, step=global_step)
```

Or with TRL's GRPOTrainer:

```python
from trl import GRPOConfig

config = GRPOConfig(
    report_to="trackio",
    trackio_space_id="your-org/qed-math-grpo",
    # ...
)
```

## Deployment

```bash
openenv push
```