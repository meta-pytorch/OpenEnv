# REPL Environment for OpenEnv

A Python REPL environment for training language models on code execution tasks, based on the [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) paradigm.

## Overview

The RLM paradigm allows language models to:
- Execute Python code in a sandboxed REPL environment
- Make recursive calls to themselves or other LMs via `llm_query()` / `llm_batch()`
- Handle near-infinite context by programmatically decomposing and exploring data
- Terminate with explicit `FINAL(answer)` or `answer = {"content": ..., "ready": True}` signals

## Features

- **Sandboxed Python Execution**: Safe code execution with restricted builtins
- **Context Loading**: Load large contexts that agents can explore programmatically
- **Multiple Finalization Patterns**:
  - Direct call: `FINAL(answer)` - helper function injected into namespace
  - Print pattern: `print('FINAL(answer)')` or `print('FINAL_VAR(var_name)')`
  - Prime Intellect style: `answer = {"content": "...", "ready": True}`
- **Iteration Limits**: Configurable maximum steps per episode
- **Reward Signals**: Customizable reward functions for RL training
- **Optional LLM Oracle**: Can enable `llm_query()` and `llm_batch()` for recursive calls

## Quick Start

```python
from repl_env import REPLEnv, REPLAction

# Start from Docker
env = REPLEnv.from_docker_image("repl-env:latest")

# Reset with context
result = env.reset(
    context="This is a large document with lots of text...",
    task_prompt="Find the word count"
)

# Execute code iteratively
result = env.execute("words = context.split()")
result = env.execute("count = len(words)")
result = env.execute("print(f'FINAL({count})')")

# Check result
print(f"Done: {result.done}")
print(f"Final Answer: {result.observation.metadata['final_answer']}")

env.close()
```

## Environment Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `REPL_CONTEXT` | Initial context to load | "" |
| `REPL_TASK_PROMPT` | Task description | "" |
| `REPL_MAX_ITERATIONS` | Max steps per episode | 30 |

## Action Space

```python
class REPLAction:
    code: str           # Python code to execute
    is_final: bool      # Whether this signals the final answer
    final_answer: str   # The final answer (if is_final=True)
```

## Observation Space

```python
class REPLObservation:
    result: CodeBlockResult    # Execution result (stdout, stderr, etc.)
    context_preview: str       # First 500 chars of context
    context_length: int        # Total context length
    available_variables: list  # Variables in namespace
    iteration: int             # Current iteration
    max_iterations: int        # Max iterations
    done: bool                 # Episode complete?
    reward: float              # Step reward
```

## Finalization Patterns

### Pattern 1: Direct FINAL() call (recommended)
```python
result = env.execute("answer = 42")
result = env.execute("FINAL(answer)")
# -> done=True, final_answer="42"
```

### Pattern 2: FINAL() via print
```python
result = env.execute("answer = 42")
result = env.execute("print(f'FINAL({answer})')")
# -> done=True, final_answer="42"
```

### Pattern 3: FINAL_VAR() for variable reference
```python
result = env.execute("my_result = 'The answer is 42'")
result = env.execute("print('FINAL_VAR(my_result)')")
# -> done=True, final_answer="The answer is 42"
```

### Pattern 4: Prime Intellect style answer dict
```python
result = env.execute("answer['content'] = '42'")
result = env.execute("answer['ready'] = True")
# -> done=True, final_answer="42"
```

## Prompts Module

The `prompts` module provides RLM-style prompts and parsing utilities:

```python
from repl_env import (
    RLM_SYSTEM_PROMPT,           # Full system prompt for capable models
    RLM_SYSTEM_PROMPT_COMPACT,   # Shorter prompt for smaller models
    build_initial_prompt,        # Build initial user prompt
    build_continuation_prompt,   # Build continuation prompt after execution
    extract_code_blocks,         # Extract Python code from LLM responses
    format_observation,          # Format observation for next LLM turn
)

# Example: Build messages for chat API
messages = [
    {"role": "system", "content": RLM_SYSTEM_PROMPT},
    {"role": "user", "content": build_initial_prompt(
        task_prompt="Count words in the context",
        context_length=1000,
        context_preview="The quick brown fox...",
        variables=["context", "answer"],
    )},
]

# Extract code from LLM response
response = "Here's my solution:\n```python\ncount = len(context.split())\nFINAL(count)\n```"
code_blocks = extract_code_blocks(response)  # ["count = len(context.split())\nFINAL(count)"]
```

## Examples

See the `examples/` directory for complete working examples:

- **`examples/repl_with_llm.py`** - Full RLM loop with Qwen models

Run example:
```bash
# LLM example (requires GPU and transformers)
python examples/repl_with_llm.py
```

## Model Usage

### Inference Loop

A typical model inference loop where the LLM generates code and the environment executes it:

```python
from repl_env import REPLEnv

# Connect to running server (or use Docker)
with REPLEnv(base_url="http://localhost:8000") as env:
    # Reset with a task
    result = env.reset(
        context="The quick brown fox jumps over the lazy dog. " * 1000,
        task_prompt="Count how many times 'fox' appears"
    )

    # Model inference loop
    while not result.done:
        # 1. Format observation for the model
        prompt = f"""You are in a Python REPL. Execute code to solve the task.
Task: {result.observation.metadata.get('task_prompt')}
Context preview: {result.observation.context_preview}...
Available variables: {result.observation.available_variables}
Iteration: {result.observation.iteration}/{result.observation.max_iterations}

Last output: {result.observation.result.stdout}

Generate Python code (end with FINAL(answer) when done):"""

        # 2. Get code from LLM
        code = your_llm.generate(prompt)

        # 3. Execute in environment
        result = env.execute(code)

        # 4. Check for errors
        if not result.observation.result.success:
            print(f"Error: {result.observation.result.exception}")

    # Episode complete
    print(f"Final answer: {result.observation.metadata.get('final_answer')}")
```

### Recursive LLM Calls (RLM Paradigm)

The key insight of RLM is that models can make recursive calls to themselves or other LLMs from within the code:

```python
from repl_env.server.repl_environment import REPLEnvironment
from repl_env.models import REPLAction

# Define LLM oracle functions
def llm_query(prompt: str) -> str:
    """Single LLM call - model can call this from executed code"""
    return your_llm.generate(prompt)

def llm_batch(prompts: list[str]) -> list[str]:
    """Batch LLM calls for efficiency"""
    return [your_llm.generate(p) for p in prompts]

# Create environment with LLM oracle
env = REPLEnvironment(
    llm_query_fn=llm_query,
    llm_batch_fn=llm_batch,
)

obs = env.reset(
    context=massive_document,  # Could be 100K+ chars
    task_prompt="Summarize each section and find key themes"
)

# The model can now generate code like this:
code = """
# Split document into sections
sections = context.split('\\n\\n')

# Use LLM to summarize each section (recursive call!)
summaries = llm_batch([f"Summarize: {s[:1000]}" for s in sections[:10]])

# Combine summaries
combined = '\\n'.join(summaries)

# Final synthesis using another LLM call
answer['content'] = llm_query(f"Find key themes in: {combined}")
answer['ready'] = True
"""

obs = env.step(REPLAction(code=code))
print(f"Done: {obs.done}, Answer: {obs.metadata['final_answer']}")
```

### RL Training Integration

For RL training, integrate with frameworks like TRL, prime-rl, or verifiers:

```python
from repl_env.server.repl_environment import REPLEnvironment
from repl_env.models import REPLAction

def collect_trajectory(env, policy, context, task):
    """Collect a single trajectory for RL training"""
    obs = env.reset(context=context, task_prompt=task)

    trajectory = []
    total_reward = 0

    while not obs.done:
        # Policy generates code (pseudo-code: replace with your LLM inference)
        code = policy.generate(obs)

        # Step environment
        next_obs = env.step(REPLAction(code=code))

        # Store transition
        trajectory.append({
            "observation": obs,
            "action": code,
            "reward": next_obs.reward,
            "next_observation": next_obs,
            "done": next_obs.done,
        })

        total_reward += next_obs.reward
        obs = next_obs

    return trajectory, total_reward

# Training loop
env = REPLEnvironment(
    reward_on_success=1.0,
    reward_on_error=-0.05,
    reward_on_failure=-0.1,
)

for epoch in range(num_epochs):
    for context, task, ground_truth in dataset:
        trajectory, reward = collect_trajectory(env, policy, context, task)

        # Verify answer correctness (optional external reward)
        # Note: assumes trajectory is non-empty; add check in production code
        final_answer = trajectory[-1]["next_observation"].metadata.get("final_answer")
        if final_answer == ground_truth:
            reward += verification_bonus

        # Update policy (pseudo-code: use your RL framework - PPO, GRPO, DPO, etc.)
        policy.update(trajectory, reward)
```

### Reward Configuration

Configure rewards for different outcomes:

```python
env = REPLEnvironment(
    reward_on_success=1.0,    # When FINAL() is called
    reward_on_iteration=0.0,  # Per step (can be negative to encourage efficiency)
    reward_on_error=-0.05,    # When code execution fails
    reward_on_failure=-0.1,   # When max iterations reached without answer
)
```

## Running Locally

### Using UV
```bash
cd envs/repl_env
uv run --project . server
```

### Using Docker
```bash
docker build -t repl-env:latest -f server/Dockerfile .
docker run -p 8000:8000 repl-env:latest
```

### Testing
```bash
pytest tests/
```

## References

- [RLM Paper (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- [RLM Implementation](https://github.com/alexzhang13/rlm)
- [Alex Zhang's RLM Blog](https://alexzhang13.github.io/blog/2025/rlm/)
- [Prime Intellect RLM Blog](https://www.primeintellect.ai/blog/rlm)
