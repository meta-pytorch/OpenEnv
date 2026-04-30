# Evaluating agents with Inspect AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/evaluation_inspect.ipynb)

After training a model in an OpenEnv environment, you need to measure how it
actually performs on a held-out set of episodes. OpenEnv integrates with
[Inspect AI](https://inspect.aisi.org.uk/) — an open-source evaluation
framework by the UK AI Safety Institute — through `InspectAIHarness`.

```{note}
**Time**: ~20 minutes | **Difficulty**: Intermediate | **GPU Required**: No
```

## How the pieces fit together

Inspect AI and OpenEnv are complementary, not overlapping:

- **OpenEnv** provides the environment (reset, step, reward) and the training
  infrastructure (GRPO via TRL).
- **Inspect AI** provides the evaluation infrastructure: datasets, solvers,
  scorers, and structured logs.

`InspectAIHarness` is the bridge. It wraps `inspect_ai.eval()` inside
OpenEnv's `EvalHarness` interface so that eval runs are tracked with the same
structured `EvalConfig` / `EvalResult` types you use across all harnesses.

The typical workflow is:

```
Train with OpenEnv (GRPO / SFT)
        ↓
Define an Inspect AI Task
  - dataset: held-out episodes or prompts
  - solver: calls your model + the OpenEnv env
  - scorer: grades correctness using env reward or exact match
        ↓
Run via InspectAIHarness → EvalResult with structured scores
```

## Install dependencies

```bash
pip install "inspect-ai>=0.3.0"
pip install "openenv-core @ git+https://github.com/meta-pytorch/OpenEnv.git"
```

`inspect-ai` is an optional dependency — `InspectAIHarness` is importable
without it, but raises a clear `ImportError` at call time if it is missing.

## Set your model provider

Pick one provider. Uncomment the corresponding lines — no other cells need to change.

```python
import os

# --- Option A: OpenAI ---
os.environ["OPENAI_API_KEY"] = "sk-..."  # replace with your key
MODEL = "openai/gpt-4o-mini"

# --- Option B: Anthropic ---
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
# MODEL = "anthropic/claude-haiku-4-5-20251001"
```

The `model` string uses `provider/model-name` format — Inspect AI extracts the
provider from the prefix and routes accordingly.

## Define an Inspect AI task for an OpenEnv environment

An Inspect AI `Task` has three parts: a **dataset** of samples to evaluate,
a **solver** that runs the model (and optionally the environment), and a
**scorer** that grades each sample.

The example below evaluates a model against `echo_env` — the reference
OpenEnv environment. The model is asked to repeat a phrase; the solver sends
the phrase to the environment and records the echoed response; the scorer
checks it matches the expected output.

The solver calls Inspect AI's `generate()` to get the model's output, then
sends it to the environment. The dataset, scorer, and harness are identical
for both providers.

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver

from openenv.core import MCPToolClient

ECHO_ENV_URL = "https://openenv-echo-env.hf.space"


@task
def openenv_echo_eval(base_url: str = ECHO_ENV_URL):
    return Task(
        dataset=[
            Sample(input="Repeat exactly: hello world", target="hello world"),
            Sample(input="Repeat exactly: inspect ai", target="inspect ai"),
            Sample(input="Repeat exactly: openenv eval", target="openenv eval"),
            Sample(input="Repeat exactly: reinforcement learning", target="reinforcement learning"),
            Sample(input="Repeat exactly: hugging face", target="hugging face"),
        ],
        solver=echo_env_solver(base_url=base_url),
        scorer=echo_scorer(),
    )


@solver
def echo_env_solver(base_url: str):
    """Ask the model to repeat the phrase, then echo it through the env."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        model_output = state.output.completion.strip()

        # echo_env is a pure MCP environment — use MCPToolClient + call_tool
        env = MCPToolClient(base_url=base_url)
        try:
            await env.reset()
            echoed = await env.call_tool("echo_message", message=model_output)
            state.metadata["echoed"] = str(echoed) if echoed is not None else ""
        finally:
            await env.close()
        return state

    return solve


@scorer(metrics=[accuracy()])
def echo_scorer():
    """CORRECT if the env echoed back exactly what the target phrase was."""

    async def score(state: TaskState, target: Target) -> Score:
        echoed = state.metadata.get("echoed", "").strip()
        expected = target.text.strip()
        return Score(
            value=CORRECT if echoed == expected else INCORRECT,
            explanation=f"Env echoed {echoed!r}, expected {expected!r}",
        )

    return score
```

```{note}
`echo_env` is a pure MCP environment. Interact with it via `MCPToolClient`
and `call_tool("echo_message", ...)`. For non-MCP environments, use
`GenericEnvClient` instead.
```

## Run the eval with `InspectAIHarness`

Pass the task to `InspectAIHarness` via `EvalConfig`. The `task` key in
`eval_parameters` takes a task object or a registered task name string.

```python
import inspect_ai
import openenv

from openenv.core.evals import EvalConfig, EvalResult, InspectAIHarness

harness = InspectAIHarness(log_dir="./eval-logs")

config = EvalConfig(
    harness_name="InspectAIHarness",
    harness_version=inspect_ai.__version__,
    library_versions={"openenv": openenv.__version__},
    dataset="openenv_echo_eval",
    eval_parameters={
        "model": MODEL,
        "task": openenv_echo_eval(base_url="https://openenv-echo-env.hf.space"),
        "temperature": 0.0,
        # The free HF Space only allows 1 concurrent WebSocket session.
        # max_connections=1 makes Inspect AI process samples sequentially.
        "max_connections": 1,
    },
)

result: EvalResult = harness.run_from_config(config)
print(result.scores)
# {'accuracy': 1.0}
```

The `EvalResult` carries both the config and the scores, making it easy to
log, compare across runs, or serialize to JSON:

```python
import json

print(json.dumps(result.model_dump(), indent=2))
```

## Using a task file instead of a task object

Inspect AI tasks can also be defined in standalone `.py` files and referenced
by path. This is useful for CI pipelines where the task definition lives in
the repo and the harness is called from a script:

```python
# tasks/echo_eval.py  (contains the @task definition above)

result = harness.run_from_config(EvalConfig(
    harness_name="InspectAIHarness",
    harness_version=inspect_ai.__version__,
    library_versions={"openenv": openenv.__version__},
    dataset="tasks/echo_eval.py@openenv_echo_eval",
    eval_parameters={
        "model": "openai/gpt-4o-mini",
        "task": "tasks/echo_eval.py@openenv_echo_eval",
    },
))
```

## Adapting to your own environment and task

Replace `echo_env_solver` with a solver that uses your env and model:

1. **Dataset** — collect held-out episodes from your env (or a static
   benchmark); each `Sample` needs `input` and `target` fields.
2. **Solver** — call your trained model against the env via `generate()`.
   If you used GRPO training with an `environment_factory`, reuse the same
   factory here so the eval env matches training exactly.
3. **Scorer** — use the env's reward signal directly, or write an Inspect AI
   `@scorer` that checks the final observation against a ground-truth target.

```python
from inspect_ai.solver import Generate, TaskState, solver
from openenv.core import MCPToolClient


@solver
def my_env_solver(base_url: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)
        model_output = state.output.completion.strip()

        env = MCPToolClient(base_url=base_url)
        try:
            await env.reset()
            result = await env.call_tool("your_tool_name", message=model_output)
            state.metadata["env_result"] = result
        finally:
            await env.close()
        return state

    return solve
```

## Next steps

- [Rubrics tutorial](rubrics) — define reward functions inside
  the environment using composable rubrics
- [Wordle GRPO](wordle-grpo) — full training loop that produces a model you
  can evaluate with this tutorial
- [Inspect AI documentation](https://inspect.aisi.org.uk/) — full reference
  for tasks, solvers, scorers, and the log viewer
