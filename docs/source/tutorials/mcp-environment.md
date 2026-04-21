# Building and Using MCP Environments

OpenEnv speaks **MCP** ([Model Context Protocol](https://modelcontextprotocol.io)) as its agent-facing interface. If an environment exposes tools, those tools live behind MCP — not behind ad-hoc REST endpoints or custom action schemas. This tutorial walks through the two sides of that contract: **consuming** an MCP environment from a training / simulation loop, and **building** one from scratch.

By the end you should be able to:

- List and call tools on any MCP environment through `step()`.
- Write a minimal `MCPEnvironment` subclass that exposes Python functions as tools.
- Understand which API path (`/ws` vs `/mcp`) belongs to which audience.

## Why MCP?

OpenEnv's dual API boundary splits responsibilities between two audiences:

- **Training / orchestration infrastructure** uses the Gym-style control plane — `reset()`, `step()`, `state()` — over WebSocket (`/ws`). This is what the trainer needs to roll out episodes, compute rewards, and enforce termination.
- **Agents** use MCP tools over the `/mcp` JSON-RPC endpoint. Tools are what the model calls to act on the world; they have discoverable schemas and deterministic names.

Standardising on MCP for agent actions means a single environment can be trained via GRPO, served for inference through an MCP-compatible client, and inspected with off-the-shelf MCP tooling — without separate interfaces to maintain.

:::{note}
In simulation mode, MCP tool calls flow **through** `step()`. The trainer stays in control of timing, rewards, and termination; the MCP action types are just a standardised action schema. The [MCP environment lifecycle guide](../mcp-environment-lifecycle.md) covers the split in depth.
:::

## Consuming an MCP Environment

The two MCP action types are `ListToolsAction` (discover what's available) and `CallToolAction` (invoke one). They behave like any other Gym action — pass them to `step()` and inspect the returned observation.

### Discovering tools

```python
from echo_env.server.echo_environment import EchoEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, ListToolsObservation

env = EchoEnvironment()
env.reset()

obs = env.step(ListToolsAction())
assert isinstance(obs, ListToolsObservation)

for tool in obs.tools:
    print(f"{tool.name}: {tool.description}")
```

Each `Tool` carries a `name`, a `description`, and an `input_schema` (JSON Schema) describing the accepted arguments. The schema is what lets a language-model agent know which parameters to fill in when it emits a tool call.

### Calling a tool

```python
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

obs = env.step(
    CallToolAction(
        tool_name="echo_message",
        arguments={"message": "Hello from MCP!"},
    )
)

assert isinstance(obs, CallToolObservation)
print(obs.tool_name)       # "echo_message"
print(obs.result.data)     # "Hello from MCP!" — the raw tool return value
print(obs.error)           # None
```

`obs.result` is a `CallToolResult` wrapper exposing the return value in a few shapes: `.data` is the raw Python value the tool returned, `.structured_content` is its JSON-encoded form, and `.content` is the MCP protocol's list of typed content blocks (useful when a tool returns rich multi-part output). `obs.error` is set only when the **framework** could not deliver the call (transport failure, unknown tool name, malformed arguments). Tool-specific failures — a business-logic error that the tool itself raised — come back inside `result`, so callers can handle them like any domain-specific response.

### Error handling

```python
obs = env.step(
    CallToolAction(tool_name="does_not_exist", arguments={}),
)

assert isinstance(obs, CallToolObservation)
print(obs.error.error_type)  # ToolErrorType.TOOL_NOT_FOUND
print(obs.error.message)     # "Unknown tool: 'does_not_exist'"
```

The `ToolError.error_type` enum (`TOOL_NOT_FOUND`, `INVALID_ARGS`, `EXECUTION_ERROR`, `TRANSPORT_ERROR`, `TIMEOUT`) lets training code distinguish between bugs in the agent, bugs in the environment, and transient infrastructure issues — which often warrant different reward signals.

### `step(CallToolAction(...))` vs `call_tool()`

Environment clients that inherit from `MCPToolClient` (such as `EchoEnv` and `FinQAEnv`) expose a shorter **async** `await env.call_tool("name", arg=value)` helper. Functionally equivalent in simulation mode — it still goes through the step loop and still updates rewards, step counts, and trajectory state — but returns the tool's raw return value directly instead of a `CallToolObservation`. Use `step(CallToolAction(...))` when you need the whole observation (reward, done, metadata); reach for `call_tool()` in async scripts where the result is all you care about. The [lifecycle guide](../mcp-environment-lifecycle.md#which-pattern-should-you-use) covers the exact trade-offs.

## Building an MCP Environment

The provider side is just as small. Subclass `MCPEnvironment`, create a `FastMCP` server, register tools with the `@mcp.tool` decorator, and pass the server to `super().__init__`. Here is the echo environment, trimmed from [`envs/echo_env/server/echo_environment.py`](https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/echo_environment.py) down to the parts this tutorial covers:

```python
from uuid import uuid4

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State


class EchoEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("echo_env")

        @mcp.tool
        def echo_message(message: str) -> str:
            """Echo back the provided message.

            Args:
                message: The message to echo back

            Returns:
                The same message that was provided
            """
            return message

        @mcp.tool
        def echo_with_length(message: str) -> dict:
            """Echo back the message with its length.

            Args:
                message: The message to echo back

            Returns:
                Dictionary with the message and its length
            """
            return {"message": message, "length": len(message)}

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        return Observation(done=False, reward=0.0, metadata={"status": "ready"})

    def _step_impl(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        # Called for non-MCP actions. Echo exposes MCP tools only,
        # so anything that isn't ListToolsAction / CallToolAction is an error.
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unsupported action type: {type(action).__name__}"},
        )

    @property
    def state(self) -> State:
        return self._state
```

A few things worth calling out:

- **Docstring → schema.** `FastMCP` inspects each tool's signature and Google-style docstring to build the `input_schema` automatically. The `Args:` block becomes parameter descriptions, and type hints become JSON types. No hand-written schemas.
- **Reserved names.** `reset`, `step`, `state`, and `close` are reserved and cannot be tool names — they belong to the infrastructure boundary. Trying to register a tool with one of those names raises at construction time.
- **`_step_impl` is required, `step` is not.** `MCPEnvironment.step` already routes `ListToolsAction` and `CallToolAction` through the FastMCP server for you. Your subclass only has to implement `_step_impl`, which the base class calls for any **non-MCP** action. In pure-MCP environments like Echo it just returns an error observation; in environments that mix tool calls with other action types (e.g. a terminal "submit" action) it's where that extra dispatch lives.
- **Rewards and `done` still work.** Because MCP actions flow through `step()`, you can compute rewards, flip `done`, and emit metadata just like in any other OpenEnv environment.

## Running the Demo End-to-End

The repo ships a self-contained walkthrough at [`examples/echo_mcp_demo.py`](https://github.com/meta-pytorch/OpenEnv/blob/main/examples/echo_mcp_demo.py). Run it directly from the repo root:

```bash
PYTHONPATH=src:envs uv run python examples/echo_mcp_demo.py
```

You will see the discovery call, two tool invocations, and an error case printed in sequence — the same four steps covered above, end-to-end against the real `EchoEnvironment`.

## Next Steps

- **MCP lifecycle details** — the [MCP Environment Lifecycle guide](../mcp-environment-lifecycle.md) covers `step()` vs `step_async()`, the `call_tool()` convenience path, and common debugging questions.
- **A richer MCP environment** — [`envs/finqa_env/`](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/finqa_env) shows tool calls participating in episode progression, rewards, and terminal submission — not just a stateless echo.
- **Design rationale** — [RFC 003](https://github.com/meta-pytorch/OpenEnv/blob/main/rfcs/003-mcp-support.md) explains why OpenEnv picked MCP as the agent boundary and how tool-calling and CodeAct styles share the same plumbing.
- **Serving tools to an external agent** — the `/mcp` JSON-RPC endpoint is available alongside `/ws` on any MCP environment server. Point an MCP-compatible client at it for production inference without going through the step loop.
