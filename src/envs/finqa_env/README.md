# FinQA Environment

A financial question-answering environment for RL training. Evaluates LLMs on their ability to answer complex financial questions using tool calls on SEC 10-K filing data.

Based on [FinQABenchmark](https://github.com/snorkel-ai/FinQABenchmark) from Snorkel AI.

## Overview

FinQA tests an agent's ability to:
- Explore available financial tables for a company
- Query table metadata and execute SQL queries
- Perform calculations on extracted data
- Submit final answers to financial questions

**Dataset**: 290 questions from SEC 10-K filings across multiple companies (Alphabet, Amazon, Apple, AT&T, etc.)

**Reward**: Binary (1.0 for correct answer, 0.0 for incorrect) using fuzzy numerical matching with 1% tolerance.

## Quick Start

### Using Docker

```bash
# Build the image (from OpenEnv repo root)
docker build -t finqa-env:latest -f src/envs/finqa_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 finqa-env:latest
```

### Using the Client

```python
from envs.finqa_env import FinQAEnv, FinQAAction

# Connect to running server
client = FinQAEnv(base_url="http://localhost:8000")

# Or start from Docker image
client = FinQAEnv.from_docker_image("finqa-env:latest")

# Reset to get a question
result = client.reset()
print(f"Question: {result.observation.question}")
print(f"Company: {result.observation.company}")

# Use tools to find the answer
# Step 1: Get available tables
result = client.step(FinQAAction(
    tool_name="get_descriptions",
    tool_args={"company_name": result.observation.company}
))
print(f"Available tables: {result.observation.tool_result}")

# Step 2: Get table info
result = client.step(FinQAAction(
    tool_name="get_table_info",
    tool_args={
        "company_name": "alphabet",
        "table_name": "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock"
    }
))

# Step 3: Query the table
result = client.step(FinQAAction(
    tool_name="sql_query",
    tool_args={
        "company_name": "alphabet",
        "table_name": "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock",
        "query": "SELECT * FROM us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock WHERE year = '2022'"
    }
))

# Step 4: Submit answer
result = client.step(FinQAAction(
    tool_name="submit_answer",
    tool_args={"answer": "6.118"}
))

print(f"Done: {result.done}")
print(f"Reward: {result.reward}")  # 1.0 if correct

client.close()
```

## Available Tools

| Tool | Description | Arguments |
|------|-------------|-----------|
| `get_descriptions` | Get list of available table names for a company | `company_name: str` |
| `get_table_info` | Get table metadata (columns, dtypes, unique values) | `company_name: str, table_name: str` |
| `sql_query` | Execute SQL query on a table (requires filters) | `company_name: str, table_name: str, query: str` |
| `submit_answer` | Submit final answer (ends episode) | `answer: str` |

### Tool Constraints

- **sql_query**: Must include filters (`WHERE`, `HAVING`, etc.). `SELECT *` is not allowed.

## Data Models

### FinQAAction

```python
@dataclass
class FinQAAction(Action):
    tool_name: str  # One of: get_descriptions, get_table_info, sql_query, submit_answer
    tool_args: Dict[str, Any]
```

### FinQAObservation

```python
@dataclass
class FinQAObservation(Observation):
    question: str           # The financial question
    company: str            # Company name
    tool_result: str        # Result of last tool call
    history: List[Dict]     # Previous tool calls and results
    step_count: int         # Current step number
    available_tools: List[str]
    done: bool              # Episode terminated?
    reward: Optional[float] # Reward (only when done=True)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FINQA_DATA_PATH` | `/app/src/envs/finqa_env/data` | Path to data directory |
| `FINQA_MAX_STEPS` | `20` | Maximum tool calls per episode |
| `FINQA_TASK` | `finqa` | Task name |

## Reward Computation

Rewards use fuzzy numerical matching:

- Extracts numbers from `\boxed{...}` format
- Handles percentages, fractions, and decimals
- 1% relative tolerance or 0.01 absolute tolerance
- Returns `1.0` for correct, `0.0` for incorrect

## Local Development

```bash
# From OpenEnv repo root
cd src/envs/finqa_env

# Run server locally
FINQA_DATA_PATH=./data uvicorn server.app:app --reload --port 8000

# Test with curl
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

## Integration with RL Frameworks

### TRL (GRPO)

```python
from trl import GRPOTrainer
from envs.finqa_env import FinQAEnv, FinQAAction

def rollout_func(prompts, trainer):
    env = FinQAEnv(base_url="http://localhost:8000")
    result = env.reset()

    # Your agent logic here
    # ...

    return {"reward": result.reward, "completion": completion}

trainer = GRPOTrainer(
    model=model,
    rollout_func=rollout_func,
    ...
)
```

## Project Structure

```
finqa_env/
├── __init__.py           # Exports FinQAAction, FinQAObservation, FinQAEnv
├── models.py             # Data models
├── client.py             # HTTP client
├── README.md             # This file
├── data/                 # Benchmark data (symlink to FinQABenchmark)
│   ├── benchmark_questions/
│   │   └── finqa.csv
│   └── input_companies/
│       └── [company folders]
└── server/
    ├── __init__.py
    ├── finqa_environment.py  # Core environment logic
    ├── tools.py              # Tool implementations
    ├── rewards.py            # Reward computation
    ├── app.py                # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```

## References

- [HuggingFace Dataset](https://huggingface.co/datasets/snorkelai/agent-finance-reasoning)
- [Leaderboard](https://leaderboard.snorkel.ai/category/snorkelfinance)
