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

Mathematical proof generation and evaluation environment for OpenEnv. Agents receive math problems, submit proofs, and receive LLM-based rubric grading (0-7 scale) with normalized rewards.

## Quick Start

```python
from qed_math_env import QEDMathEnv

# Connect to environment
with QEDMathEnv(base_url="http://localhost:8000") as env:
    # Reset to load a problem
    obs = await env.reset()
    print(f"Problem: {obs.problem[:100]}...")
    print(f"Rubric: {obs.grading_guidelines[:100]}...")
    
    # Submit proof
    result = await env.submit_proof(proof="By induction on n...")
    print(f"Score: {result.score}/7")
    print(f"Reward: {result.reward:.2f}")
```

## Project Structure

```
qed_math_env/
├── __init__.py              # Module exports
├── models.py                # Action/Observation types
├── client.py               # QEDMathEnv client
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Dependencies
├── README.md               # This file
└── server/
    ├── __init__.py
    ├── qed_math_environment.py  # Main environment
    ├── mcp_server.py           # MCP tools
    ├── rubric.py               # Grading rubric
    ├── app.py                  # FastAPI server
    └── Dockerfile              # Container
```

## Data Format

### Input (from dataset)

```json
{
  "problem": "Prove that for any integer n...",
  "solution": "Proof goes here...",
  "rubrics": [
    {"title": "Base Case", "points": 2, "desc": "..."},
    {"title": "Inductive Step", "points": 3, "desc": "..."},
    {"title": "Conclusion", "points": 2, "desc": "..."}
  ],
  "dataset": "FineProofs-RL"
}
```

### Observation (ProblemObservation)

- `problem`: Math problem statement
- `reference_solution`: Ground truth solution
- `grading_guidelines`: Rubric (0-7 scale)
- `problem_id`: Unique identifier
- `dataset_source`: Source dataset name

### Action (SubmitProof)

- `proof`: Agent's proof submission

### Result (ProofSubmissionObservation)

- `score`: Grade (0-7)
- `feedback`: Grader feedback
- `reward`: Normalized (score/7)
- `done`: True after submission

## Reward Computation

Standard: `reward = score / 7.0`

With thresholding (sparse reward):
- score < 1 → reward = score / 7.0
- 1 <= score < 6 → reward = 1.0
- score >= 6 → reward = score / 7.0

## Building

```bash
# Build Docker image
docker build -t qed-math-env:latest -f server/Dockerfile .

# Run locally
uvicorn server.app:app --reload --port 8000
```

## Deployment

```bash
openenv push
```