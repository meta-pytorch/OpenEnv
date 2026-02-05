---
title: REPL Environment Server
emoji: 🎮
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# REPL Environment for OpenEnv

A Python REPL environment for training language models on code execution tasks, based on the [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) paradigm.

## Overview

The RLM paradigm allows language models to:
- Execute Python code in a sandboxed REPL environment
- Make recursive calls to themselves or other LMs via `llm_query()` / `llm_query_batched()`
- Handle near-infinite context by programmatically decomposing and exploring data
- Terminate with explicit `FINAL(answer)` or `answer = {"content": ..., "ready": True}` signals

## Features

- **Unified API**: Same `REPLEnv` class works for both local and remote execution
- **Sandboxed Python Execution**: Safe code execution with restricted builtins
- **Context Loading**: Load large contexts that agents can explore programmatically
- **Multiple Finalization Patterns**:
  - Direct call: `FINAL(answer)` - helper function injected into namespace
  - Print pattern: `print('FINAL(answer)')` or `print('FINAL_VAR(var_name)')`
  - Prime Intellect style: `answer = {"content": "...", "ready": True}`
- **Iteration Limits**: Configurable maximum steps per episode
- **Reward Signals**: Customizable reward functions for RL training
- **Optional LLM Oracle**: Can enable `llm_query()` and `llm_query_batched()` for recursive calls

## Quick Start

### Local Mode (No Server Required)

```python
from repl_env import REPLEnv

# Create environment - runs locally by default
with REPLEnv() as env:
    result = env.reset(
        context="This is a large document with lots of text...",
        task_prompt="Find the word count"
    )

    # Execute code iteratively
    result = env.execute("words = context.split()")
    result = env.execute("count = len(words)")
    result = env.execute("print(f'FINAL({count})')")

    print(f"Done: {result.done}")
    print(f"Final Answer: {env.state().final_answer}")
```

### Remote Server Mode

```python
from repl_env import REPLEnv

# Connect to a running server - same API!
with REPLEnv(base_url="https://my-server.hf.space") as env:
    result = env.reset(context="...", task_prompt="...")
    result = env.execute("count = len(context)")
    result = env.execute("print(f'FINAL({count})')")
```

### Local Mode with LLM Support

```python
from repl_env import REPLEnv

def my_llm_query(prompt: str) -> str:
    return your_llm.generate(prompt)

def my_llm_query_batched(prompts: list[str]) -> list[str]:
    return [my_llm_query(p) for p in prompts]

# Pass LLM functions for recursive calls
with REPLEnv(llm_query_fn=my_llm_query, llm_batch_fn=my_llm_query_batched) as env:
    result = env.reset(context=large_document, task_prompt="Summarize this")

    # Now the executed code can use llm_query() and llm_query_batched()!
    result = env.execute("summary = llm_query('Summarize: ' + context[:1000])")
```

### From Docker or HuggingFace Hub

```python
from repl_env import REPLEnv

# Start from Docker image
env = REPLEnv.from_docker_image("repl-env:latest")

# Or from HuggingFace Hub
env = REPLEnv.from_hub("openenv/repl-env")
```

## API Reference

### REPLEnv

```python
class REPLEnv:
    def __init__(
        self,
        base_url: str | None = None,      # Server URL (None = local mode)
        *,
        # Local-only options
        llm_query_fn: Callable | None = None,    # Function for llm_query()
        llm_batch_fn: Callable | None = None,    # Function for llm_query_batched()
        max_output_length: int = 8192,           # Max stdout/stderr chars
        context_preview_length: int = 500,       # Chars in context preview
        reward_on_success: float = 1.0,          # Reward on FINAL()
        reward_on_iteration: float = 0.0,        # Reward per step
        reward_on_failure: float = -0.1,         # Reward on max iterations
        reward_on_error: float = -0.05,          # Reward on execution error
        # Remote-only options
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ): ...

    def reset(
        self,
        *,
        context: str = "",              # Text to analyze (as `context` variable)
        task_prompt: str = "",          # Task description
        max_iterations: int = 30,       # Max code execution steps
        seed: int | None = None,        # Random seed
        episode_id: str | None = None,  # Custom episode ID
        hf_token: str | None = None,    # HF token for llm_query (remote mode)
        llm_model: str | None = None,   # Model for llm_query (remote mode)
    ) -> StepResult[REPLObservation]: ...

    def execute(self, code: str) -> StepResult[REPLObservation]: ...
    def step(self, action: REPLAction) -> StepResult[REPLObservation]: ...
    def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]: ...
    def state(self) -> REPLState: ...
    def close(self) -> None: ...
```

### Action Space

```python
class REPLAction:
    code: str = ""                    # Python code to execute
    is_final: bool = False            # Whether this signals the final answer
    final_answer: str | None = None   # The final answer (if is_final=True)
```

### Observation Space

```python
class REPLObservation:
    result: CodeBlockResult      # Execution result (stdout, stderr, etc.)
    context_preview: str | None  # First 500 chars of context
    context_length: int          # Total context length
    available_variables: list    # Variables in namespace
    iteration: int               # Current iteration
    max_iterations: int          # Max iterations
    done: bool                   # Episode complete?
    reward: float                # Step reward
    metadata: dict               # Additional info (final_answer, etc.)
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
# Direct call (recommended) - pass variable name as string
# FINAL_VAR looks up the variable and returns FINAL(value)
result = env.execute('FINAL_VAR("my_result")')
# -> done=True, final_answer="The answer is 42"

# Also works via print (for regex detection)
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
from repl_env.prompts import (
    # System prompts (from official RLM repo)
    RLM_SYSTEM_PROMPT,           # Base prompt with llm_query_batched
    RLM_SYSTEM_PROMPT_QWEN,      # For Qwen models (adds cost warning)

    # Prompt building
    QueryMetadata,               # Context metadata dataclass
    build_rlm_system_prompt,     # Build system messages with metadata
    build_user_prompt,           # Build user prompt for each iteration
    build_initial_prompt,        # Convenience wrapper for iteration 0

    # Parsing utilities
    extract_code_blocks,         # Extract code from ```repl``` or ```python``` blocks
    format_observation,          # Format execution result for LLM
)

# Example: Build messages using official RLM style
query_metadata = QueryMetadata(
    context_lengths=[len(context)],
    context_total_length=len(context),
    context_type="str",
)
messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT_QWEN, query_metadata)
messages.append(build_user_prompt(root_prompt="Count words in the context", iteration=0))

# Extract code from LLM response (supports ```repl``` and ```python```)
response = "Here's my solution:\n```repl\ncount = len(context.split())\nFINAL(count)\n```"
code_blocks = extract_code_blocks(response)  # ["count = len(context.split())\nFINAL(count)"]
```

## Examples

See the `examples/` directory for complete working examples:

- **`examples/repl_with_llm.py`** - Full RLM loop with local Qwen model
- **`examples/repl_oolong_simple.py`** - RLM on Oolong benchmark with HuggingFace Inference API

Run examples:
```bash
# Full RLM example with local model (requires GPU)
python examples/repl_with_llm.py

# Oolong benchmark with HF Inference API (requires HF_TOKEN)
python examples/repl_oolong_simple.py
```

## Model Usage

### Inference Loop

A typical model inference loop where the LLM generates code and the environment executes it:

```python
from repl_env import REPLEnv
from repl_env.prompts import RLM_SYSTEM_PROMPT, build_initial_prompt, extract_code_blocks, format_observation

# Works with both local and remote!
with REPLEnv(base_url="http://localhost:8000") as env:  # or REPLEnv() for local
    result = env.reset(
        context="The quick brown fox jumps over the lazy dog. " * 1000,
        task_prompt="Count how many times 'fox' appears"
    )

    messages = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(
            task_prompt="Count how many times 'fox' appears",
            context_length=result.observation.context_length,
            context_preview=result.observation.context_preview,
            variables=result.observation.available_variables,
        )},
    ]

    while not result.done:
        # Get code from LLM
        response = your_llm.chat(messages)
        code_blocks = extract_code_blocks(response)

        for code in code_blocks:
            result = env.execute(code)
            if result.done:
                break

        # Update conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": format_observation(result.observation)})

    print(f"Final answer: {env.state().final_answer}")
```

### Recursive LLM Calls (RLM Paradigm)

The key insight of RLM is that models can make recursive calls to themselves or other LLMs from within the code:

```python
from repl_env import REPLEnv

def llm_query(prompt: str) -> str:
    """Single LLM call - model can call this from executed code"""
    return your_llm.generate(prompt)

def llm_query_batched(prompts: list[str]) -> list[str]:
    """Batch LLM calls for efficiency (parallel in production)"""
    return [your_llm.generate(p) for p in prompts]

# Create environment with LLM oracle (local mode)
with REPLEnv(llm_query_fn=llm_query, llm_batch_fn=llm_query_batched) as env:
    result = env.reset(
        context=massive_document,  # Could be 100K+ chars
        task_prompt="Summarize each section and find key themes"
    )

    # The model can now generate code like this:
    code = """
# Split document into sections
sections = context.split('\\n\\n')

# Use LLM to summarize each section (recursive call!)
summaries = llm_query_batched([f"Summarize: {s[:1000]}" for s in sections[:10]])

# Combine summaries
combined = '\\n'.join(summaries)

# Final synthesis using another LLM call
answer['content'] = llm_query(f"Find key themes in: {combined}")
answer['ready'] = True
"""

    result = env.execute(code)
    print(f"Done: {result.done}, Answer: {env.state().final_answer}")
```

### RL Training Integration

For RL training, integrate with frameworks like TRL, prime-rl, or verifiers:

```python
from repl_env import REPLEnv

def collect_trajectory(env, policy, context, task):
    """Collect a single trajectory for RL training"""
    result = env.reset(context=context, task_prompt=task)

    trajectory = []
    total_reward = 0

    while not result.done:
        # Policy generates code
        code = policy.generate(result.observation)

        # Step environment
        next_result = env.execute(code)

        # Store transition
        trajectory.append({
            "observation": result.observation,
            "action": code,
            "reward": next_result.reward,
            "next_observation": next_result.observation,
            "done": next_result.done,
        })

        total_reward += next_result.reward
        result = next_result

    return trajectory, total_reward

# Training loop
with REPLEnv(
    reward_on_success=1.0,
    reward_on_iteration=0.0,
    reward_on_error=-0.05,
    reward_on_failure=-0.1,
) as env:
    for epoch in range(num_epochs):
        for context, task, ground_truth in dataset:
            trajectory, reward = collect_trajectory(env, policy, context, task)

            # Verify answer correctness (optional external reward)
            if trajectory:
                final_answer = env.state().final_answer
                if final_answer == ground_truth:
                    reward += verification_bonus

            # Update policy (use your RL framework - PPO, GRPO, DPO, etc.)
            policy.update(trajectory, reward)
```

### Reward Configuration

Configure rewards for different outcomes:

```python
env = REPLEnv(
    reward_on_success=1.0,    # When FINAL() is called
    reward_on_iteration=0.0,  # Per step (can be negative to encourage efficiency)
    reward_on_error=-0.05,    # When code execution fails
    reward_on_failure=-0.1,   # When max iterations reached without answer
)
```

## Environment Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `REPL_CONTEXT` | Initial context to load | "" |
| `REPL_TASK_PROMPT` | Task description | "" |
| `REPL_MAX_ITERATIONS` | Max steps per episode | 30 |
| `HF_TOKEN` | HuggingFace token for llm_query (server fallback) | None |
| `LLM_MODEL` | Model for llm_query/llm_query_batched | Qwen/Qwen3-Coder-480B-A35B-Instruct |

## Running the Server

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
