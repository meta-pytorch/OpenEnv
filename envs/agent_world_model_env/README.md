# Agent World Model

AgentWorldModel-1K is a synthetic agentic environment suite containing **1,000 tool-use environments** with **10,000 tasks** large-scale RL training. Each environment is a fully functional MCP server with tools, database state, and verification logic.


## Quick Start

### 1. Start the Server

```bash
# From the OpenEnv root directory
uv run uvicorn agent_world_model_env.server.app:app --host 127.0.0.1 --port 8000
```

### 2. Connect with the Client

```python
import asyncio
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

async def main():
    async with AWMEnv(base_url="http://localhost:8000") as env:
        # Reset to a scenario with a specific task
        result = await env.reset(scenario="marketplace_1", task_idx=0)
        print(f"Task: {result.observation.task}")
        print(f"Tools available: {result.observation.num_tools}")
        print(f"Verifier support: {result.observation.has_verifier}")  # {sql: True, code: True}

        # List available tools
        tools = await env.list_tools()
        for tool in tools[:3]:
            print(f"  - {tool.name}: {tool.description}")

        # Call a tool
        obs = await env.call_tool("sub_env_get_current_user_profile")
        print(f"Result: {obs.tool_result}")

        # Run verification (can be called multiple times with different modes)
        result = await env.step(CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": "optional answer"}
        ))
        print(f"Reward type: {result.observation.reward_type}")
        print(f"Reward: {result.reward}")  # Note: use result.reward, not result.observation.reward
        print(f"Verify result: {result.observation.verify_result}")

        # End episode (destroys environment, does not run verifier)
        result = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        print(f"Episode done: {result.done}")  # Note: use result.done to destroy the environment, if you want to keep the environment/database state/records, set keep_session=True

asyncio.run(main())
```

## Environment Details

### Actions

AWM supports two action types:

| Action | Description |
|--------|-------------|
| `ListToolsAction()` | List all available MCP tools for the current scenario |
| `CallToolAction(tool_name, arguments)` | Call a specific tool with arguments |

Special tool names:
- `"verify"` - Run verifier with `{verifier_mode: "sql"|"code", final_answer: "optional"}` arguments
- `"done"` - End the episode and destroy environment (does NOT run verifier)
- `"__list_scenarios__"` - List all 1,000 available scenarios and their tasks

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `reward` | float | Reward value based on reward_type and config |
| `reward_type` | str | Outcome classification (see below) |
| `scenario` | str | Current scenario name |
| `task` | str | Task description in natural language |
| `task_idx` | int | Task index (0-9) |
| `has_verifier` | dict/None | Verifier support: `{sql: bool, code: bool}` or None |
| `num_tools` | int | Number of tools available |
| `tool_name` | str | Name of the tool called |
| `tool_result` | Any | Result from the tool call |
| `error` | str | Error message if any |
| `verify_result` | dict | Verification output after calling verify |

### Reward Types and Values

Default reward configuration:

| Type | Reward | Description |
|------|--------|-------------|
| `complete` | 1.0 | Task completed successfully (verifier passed) |
| `incomplete` | 0.1 | Task not completed (verifier failed) |
| `format_error` | -1.0 | Format error (maps from tool_not_found, invalid_args) |
| `tool_not_found` | -1.0 | Tool name not recognized |
| `invalid_args` | -1.0 | Tool arguments invalid |
| Other types | 0.0 | server_error, timeout, etc. |

You can customize rewards at reset:
```python
result = await env.reset(
    scenario="marketplace_1",
    task_idx=0,
    reward_config={"complete": 2.0, "incomplete": 0.5, "format_error": -2.0}
)
```

## Verifier Modes

AWM supports two verification modes, selected when calling the `verify` tool:

### Code Mode (Default, no LLM needed)

```python
result = await env.step(CallToolAction(
    tool_name="verify",
    arguments={"verifier_mode": "code", "final_answer": "optional answer"}
))
```

Executes a Python verifier function that compares initial and final database states. Deterministic and does not require LLM.

### SQL Mode (code-augmented LLM-as-a-Judge)

This mode is recommended for judge performance. You need to set the LLM credentials via environment variables before using this mode.

```python
# Set LLM credentials via environment variables
# OPENENV_AWM_LLM_BASE_URL, OPENENV_AWM_LLM_API_KEY, OPENENV_AWM_LLM_MODEL

result = await env.step(CallToolAction(
    tool_name="verify",
    arguments={"verifier_mode": "sql"}
))
```

Runs SQL queries to extract state changes, then uses an LLM judge to determine success.

### Checking Verifier Support at Reset

After reset, `has_verifier` returns a dict showing which modes are available:

```python
result = await env.reset(scenario="marketplace_1", task_idx=0)
if result.observation.has_verifier:
    print(f"SQL verifier: {result.observation.has_verifier.get('sql', False)}")
    print(f"Code verifier: {result.observation.has_verifier.get('code', False)}")
else:
    print("No verifiers available for this task")
```

## Listing Scenarios & Tasks

```python
async with AWMEnv(base_url="http://localhost:8000") as env:
    # List all 1,000 scenarios
    result = await env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))

    print(f"Total scenarios: {result.observation.total}")
    for scenario in result.observation.scenarios[:5]:
        print(f"  - {scenario['name']}: {scenario['num_tasks']} tasks")
        print(f"    Sample task: {scenario['tasks'][0][:80]}...")
```

## Building an Agent

See [`example_usage.py`](example_usage.py) for a complete example of an LLM-powered agent that:

1. Discovers available tools via `list_tools`
2. Iteratively calls tools to accomplish the task
3. Runs verification via `verify` tool (can use "sql" or "code" mode)
4. Ends episode via `done` action

```bash
# Set your LLM credentials
export ENDPOINT_URL="https://your-azure-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"

# Run the agent
python envs/agent_world_model_env/example_usage.py
```

## Project Structure

```
agent_world_model_env/
├── __init__.py              # Module exports (AWMEnv, AWMObservation, etc.)
├── README.md                # This file
├── example_usage.py         # Complete LLM agent example
├── client.py                # AWMEnv client implementation
├── models.py                # AWMAction, AWMObservation models
├── pyproject.toml           # Package dependencies
└── server/
    ├── __init__.py          # Server module exports
    ├── app.py               # FastAPI application
    ├── awm_environment.py   # Core AWMEnvironment implementation
    ├── data_loader.py       # HuggingFace dataset loader
    ├── db_manager.py        # SQLite database management
    ├── scenario_manager.py  # Sub-environment process management
    └── verifier.py          # Verification logic (code & SQL modes)
```

## Parallel RL Training

AWM is designed for large-scale agentic RL training. A single server supports **concurrent sessions**, allowing multiple agents to interact with different scenarios simultaneously.

### Server (Single Instance)

```bash
# Start one server that handles all connections
uv run uvicorn agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899 --workers 32
```

### Parallel Rollout Collection

```python
import asyncio
from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

async def run_episode(scenario: str, task_idx: int, agent_id: int):
    """Each agent connects to the same server with its own isolated session."""
    async with AWMEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(scenario=scenario, task_idx=task_idx)
        task = result.observation.task

        # Discover tools
        tools_result = await env.step(ListToolsAction())
        tools = tools_result.observation.tools

        trajectory = []
        for step in range(max_steps):
            # Your policy selects an action (tool call) based on observation
            tool_name, arguments = policy(task, tools, trajectory)

            result = await env.step(CallToolAction(
                tool_name=tool_name, arguments=arguments
            ))
            trajectory.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": result.observation.tool_result,
                "reward": result.reward,
            })

            if not tool_name:
                # if there is no more tool call, end the episode
                break
        
        result = await env.step(CallToolAction(tool_name="done", arguments={"keep_session": True}))

        # Get final reward from verifier
        result = await env.step(CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code"}
        ))
        final_reward = result.reward

        # End episode
        await env.step(CallToolAction(tool_name="done", arguments={}))

        return {"scenario": scenario, "trajectory": trajectory, "reward": final_reward}

async def collect_rollouts(
    all_tasks: list[tuple[str, int]],  # [(scenario, task_idx), ...]
    batch_size: int = 64,
    rollouts_per_task: int = 16,
    max_concurrent: int = 1024,
):
    """Collect parallel rollouts in batches for RL training.

    Each batch samples `batch_size` (scenario, task_idx) pairs, and runs
    `rollouts_per_task` independent episodes per pair for advantage estimation.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_episode(scenario, task_idx):
        async with semaphore:
            return await run_episode(scenario, task_idx)

    all_rollouts = []
    for epoch in range(num_epochs):
        # Sample a batch of (scenario, task_idx) pairs
        batch = random.sample(all_tasks, k=batch_size)

        # Launch batch_size * rollouts_per_task episodes in parallel
        coros = [
            bounded_episode(scenario, task_idx)
            for scenario, task_idx in batch
            for _ in range(rollouts_per_task)
        ]
        rollouts = await asyncio.gather(*coros)

        # Group by (scenario, task_idx) → list of rollouts for GRPO / reward shaping
        grouped = group_by_task(rollouts)  # {(scenario, task_idx): [rollout, ...]}
        all_rollouts.append(grouped)

        # ... feed grouped rollouts to your RL optimizer ...

    return all_rollouts
```

More details can be found at:

| Resource | Link |
|----------|------|
| 📄 Paper | [📄 arxiv.org/abs/2602.10090](https://arxiv.org/abs/2602.10090) |
| 💻 Synthesis Pipeline Code | [💻 Snowflake-Labs/agent-world-model](https://github.com/Snowflake-Labs/agent-world-model) |
| 📦 AgentWorldModel-1K | [🤗 Snowflake/AgentWorldModel-1K](https://huggingface.co/datasets/Snowflake/AgentWorldModel-1K) |
| 🤖 Arctic-AWM-4B | [🤗 Snowflake/Arctic-AWM-4B](https://huggingface.co/Snowflake/Arctic-AWM-4B) |
| 🤖 Arctic-AWM-8B | [🤗 Snowflake/Arctic-AWM-8B](https://huggingface.co/Snowflake/Arctic-AWM-8B) |
| 🤖 Arctic-AWM-14B | [🤗 Snowflake/Arctic-AWM-14B](https://huggingface.co/Snowflake/Arctic-AWM-14B) |


## Citation

If you find this work useful, please kindly cite:

```bibtex
@article{wang2026agentworldmodelinfinity,
      title={Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning}, 
      author={Zhaoyang Wang and Canwen Xu and Boyi Liu and Yite Wang and Siwei Han and Zhewei Yao and Huaxiu Yao and Yuxiong He},
      year={2026},
      eprint={2602.10090},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.10090}, 
}
```
