---
name: agentkernel
description: >
  Spawn and orchestrate agents as local subprocesses or Kubernetes pods.
  Each agent runs with an independent runtime, conversation, tools,
  and skills. Use when a task benefits from parallel work, role
  specialization, persistent agent state, or sandboxed execution.
metadata:
  version: "2.1"
  pre-condition: "0"
---

# AgentKernel

Spawn and orchestrate agents from `<helpers>` blocks. Each agent runs in its own process (local subprocess or Kubernetes pod) with an independent runtime, conversation state, tools, and skills. You decide what agents to create and what to say to them. The kernel handles process lifecycle, networking, image management, and health checks.

## AgentKernel vs `agents` Skill

| | `agents` skill | `agentkernel` |
|---|---|---|
| **Backends** | Local subprocesses only | Local subprocesses or k8s pods |
| **Addressing** | By name (`call_async("my-agent", ...)`) | By UUID + secret nonce |
| **Protocol** | Anthropic Messages API | Custom SSE (TurnRequest/TurnResponse) |
| **Access control** | Open — any caller can talk to any agent | Nonce-secured single-owner |
| **Teams / capacity** | No | Yes |
| **Image packaging** | No | Yes (OCI images for k8s) |
| **AgentBus** | No | Yes |
| **Dependencies** | API server skill | None |

**Use `agents`** for lightweight local agent workflows where convenience matters — create by name, call by name, check event logs.

**Use `agentkernel`** when you need k8s deployment, container isolation, capacity management, nonce-secured access, or agentbus observability.

## Core Concepts

**Kernel**: `AgentKernel` is the entry point. It wires together the spawner, agent client, and storage. The backend determines where agents run.

**Backends**: Two backends are available:
- **local** — agents run as subprocesses on the same machine. No isolation, no config file needed. Good for development and quick experiments.
- **kubernetes** — agents run as pods in a k8s cluster. Full container isolation. Requires a config file with cluster details.

The entire API after initialization is identical across backends.

**SpawnRequest + SpawnInfo**: A `SpawnRequest` defines the agent identity (name, team, metadata). The `spawn_info` field carries agent-type-specific config (system prompt, model, tools, etc.) — e.g. `OpenClawSpawnInfo`.

**Nonce**: Each spawn returns a `SpawnResult` containing the agent record and a secret nonce. The nonce is required for all communication — it enforces single-owner access. Only the entity that spawned an agent can talk to it.

**AgentBus**: Optional observability/safety layer. When enabled, all LLM inference and code execution events are logged to an agent bus that can be inspected externally.

**Teams**: Logical groups with capacity limits. Spawning into a full team raises an error.

## Initialization

### Local backend

No config file needed. Agents run as subprocesses with the same permissions as the parent process.

<helpers>
from agentic.kernel import AgentKernel
from agentic.kernel.plugins.openclaw import OpenClawPlugin

kernel = AgentKernel(backend="local", plugins=[OpenClawPlugin()])
</helpers>

### Kubernetes backend

Agents run as pods in the `agentkernel-0` namespace. The config file is at `agentkernel/examples/agentkernel.yaml`:

```yaml
backend: kubernetes
namespace: agentkernel-0
base_image: your-registry.example.com/agentkernel:latest
kubeconfig: ~/.kube/config
registry_url: your-registry.example.com
debug: true
```

- `debug: true` preserves pods on failure for inspection (otherwise they are cleaned up automatically).

<helpers>
from agentic.kernel import AgentKernel
from agentic.kernel.plugins.openclaw import OpenClawPlugin

kernel = AgentKernel.from_config("agentkernel/examples/agentkernel.yaml", plugins=[OpenClawPlugin()])
</helpers>

## API

All API calls below work identically regardless of backend.

### Spawn an Agent

<helpers>
import os
from agentic.kernel import SpawnRequest
from agentic.kernel.plugins.openclaw import OpenClawSpawnInfo

result = await kernel.spawner.spawn(SpawnRequest(
    name="researcher",
    agent_type="openclaw",
    metadata={"role": "research"},
    spawn_info=OpenClawSpawnInfo(
        system_prompt="You are a research specialist. Be thorough and cite sources.",
        model="claude-sonnet-4-5",
        api_key=os.environ.get("LLM_API_KEY", ""),
    ),
))

agent = result.agent   # Agent(id, name, team_id, state, metadata, ...)
nonce = result.nonce    # Secret — required for all communication
print(f"Spawned: {agent.id} ({agent.name})")
</helpers>

`SpawnRequest` fields:
- `name` — agent name (also used in k8s pod naming)
- `team_id` — team for capacity tracking (optional, default: "")
- `metadata` — arbitrary labels for discovery (e.g. `{"role": "worker"}`)
- `image_id` — custom image from packaging (optional, defaults to base_image in k8s)
- `spawn_info` — agent-type-specific config (e.g. `OpenClawSpawnInfo`)
- `env` — extra environment variables forwarded to the agent process

`OpenClawSpawnInfo` fields:
- `system_prompt` — system prompt for the agent
- `model` — LLM model name (default: `"claude-sonnet-4-5"`)
- `provider` — LLM provider (default: `"anthropic"`)
- `tools` — list of tool names to enable (default: `["bash"]`)
- `thinking_level` — thinking level: `"none"`, `"low"`, `"medium"`, `"high"`
- `api_key` — LLM API key (also forwarded from host `LLM_API_KEY` env var)
- `base_url` — override LLM API base URL

### Send a Message (Turn)

Use the `ask()` helper to send a message and get the full response:

<helpers>
response = await ask(kernel, agent.id, nonce, "What are the latest findings on topic X?")
print(response)
</helpers>

The agent maintains conversation state — subsequent turns see the full history.

For manual streaming (e.g. to display progress), use `kernel.agent_client.turn()` directly — note `end=""` to avoid extra newlines between tokens:

<helpers>
import json
from agentic.kernel import TurnRequest

request = TurnRequest(
    agent_id=agent.id,
    nonce=nonce,
    body=json.dumps({
        "messages": [{"role": "user", "content": "What are the latest findings on topic X?"}]
    }).encode(),
)

response_text = []
async for chunk in kernel.agent_client.turn(request):
    if chunk.body:
        print(chunk.body, end="", flush=True)
        response_text.append(chunk.body)
    if chunk.error:
        print(f"\nError: {chunk.error}")
full_response = "".join(response_text)
</helpers>

### Get History

<helpers>
history = await kernel.agent_client.get_history(agent.id, last_n=5)
for entry in history:
    print(f"[{entry['role']}] {entry['content'][:100]}")
</helpers>

### Get Agent Info

<helpers>
info = await kernel.agent_client.get_info(agent.id)
print(f"pid={info['pid']}  cwd={info['cwd']}  uid={info['uid']}")
</helpers>

### Check Status

<helpers>
statuses = await kernel.status()
for s in statuses:
    line = f"{s['name']}: state={s['state']} live={s['live']}"
    if s.get('pod_phase'):       # k8s backend
        line += f" pod={s['pod_phase']}"
    if s.get('process_alive') is not None:  # local backend
        line += f" process_alive={s['process_alive']}"
    print(line)
</helpers>

### Kill an Agent

<helpers>
await kernel.spawner.kill(agent.id)
</helpers>

### Clean Up All Agents

<helpers>
await kernel.cleanup()
</helpers>

## Teams

Teams reserve capacity and group agents together.

<helpers>
from agentic.kernel import CreateTeamRequest

# Reserve capacity
await kernel.spawner.create_team(CreateTeamRequest(
    team_id="analysis-team",
    resources={"cpu": 4},
))

# Spawn into the team
result = await kernel.spawner.spawn(SpawnRequest(
    name="analyst",
    team_id="analysis-team",
    agent_type="openclaw",
    spawn_info=OpenClawSpawnInfo(
        system_prompt="You are a data analyst.",
        api_key=os.environ.get("LLM_API_KEY", ""),
    ),
))

# Delete team (kills all agents first)
await kernel.spawner.delete_team("analysis-team")
</helpers>

## AgentBus

AgentBus adds observability and safety to agent execution. When enabled, the agent logs all LLM inference and code execution events to a bus that can be inspected via the agentbus CLI.

<helpers>
from agentic.kernel import AgentBusConfig

result = await kernel.spawner.spawn(SpawnRequest(
    name="worker",
    agent_type="openclaw",
    spawn_info=OpenClawSpawnInfo(
        system_prompt="You are a helpful worker.",
        api_key=os.environ.get("LLM_API_KEY", ""),
    ),
    agentbus=AgentBusConfig(
        port=8095,
        disable_safety=False,
    ),
))
</helpers>

To inspect the bus, you can use the agentbus skill.

# Kubernetes backend — port-forward first
kubectl --kubeconfig ~/.kube/config \
    -n agentkernel-0 port-forward pod/agent-<id-prefix> 8095:8095
# Then poll as above
```

The bus ID follows the pattern `{agent_name}.{agent_uuid}`.

## Patterns

### Fan-out / Fan-in

Spawn specialists, query them in parallel, synthesize results.

<helpers>
import asyncio

# Spawn specialists
researcher_r = await kernel.spawner.spawn(SpawnRequest(
    name="researcher", agent_type="openclaw", spawn_info=OpenClawSpawnInfo(
        system_prompt="You are a research specialist.",
        api_key=os.environ.get("LLM_API_KEY", ""),
    ),
))
analyst_r = await kernel.spawner.spawn(SpawnRequest(
    name="analyst", agent_type="openclaw", spawn_info=OpenClawSpawnInfo(
        system_prompt="You are a data analyst.",
        api_key=os.environ.get("LLM_API_KEY", ""),
    ),
))

# Fan out — ask() collects streaming chunks into a single string
research_task = asyncio.create_task(
    ask(kernel, researcher_r.agent.id, researcher_r.nonce, "Find papers on quantum error correction")
)
analysis_task = asyncio.create_task(
    ask(kernel, analyst_r.agent.id, analyst_r.nonce, "Run cost-benefit analysis on approach X")
)
research, analysis = await asyncio.gather(research_task, analysis_task)

print(f"Research: {research[:200]}")
print(f"Analysis: {analysis[:200]}")
</helpers>

### Pipeline

One agent's output feeds the next.

<helpers>
raw_data = await ask(kernel, researcher_r.agent.id, researcher_r.nonce, "Gather data on topic X")
analysis = await ask(kernel, analyst_r.agent.id, analyst_r.nonce, f"Analyze this data:\n{raw_data}")
print(analysis)
</helpers>

### Image Packaging

Bundle custom code into agent images. On local backend, bundles are copied to a directory. On k8s, an OCI image is built and pushed to the registry.

<helpers>
from agentic.kernel import SourceBundle

# Upload code to blob storage
helpers_uri = kernel.blob_store.upload_dir("./my_helpers/")

# Build an agent image with the bundle
job = await kernel.packaging.create_agent_image(
    name="custom-worker",
    bundles=[SourceBundle(uri=helpers_uri, labels={"name": "my_helpers"})],
)
if job.image:
    # Spawn an agent using the custom image
    result = await kernel.spawner.spawn(SpawnRequest(
        name="custom-agent",
        agent_type="openclaw",
        image_id=job.image.id,
        spawn_info=OpenClawSpawnInfo(
            system_prompt="You have custom tools available.",
            api_key=os.environ.get("LLM_API_KEY", ""),
        ),
    ))
</helpers>

## Lifecycle

- Agents persist (as subprocesses or pods) until explicitly killed. Always clean up when done.
- Each agent has one conversation and one owner. The nonce enforces this — only the spawner can communicate with its agent.
- Teams have capacity limits. Spawning into a full team raises `ValueError`.
- The `LLM_API_KEY` and `OPENAI_API_KEY` environment variables are automatically forwarded to agent processes.

## Operations

**Note**: If behind a proxy, configure `HTTP_PROXY`/`HTTPS_PROXY` environment variables.

### Run examples locally

```bash
# Single agent, local backend (no config file needed)
LLM_API_KEY=... uv run python -m agentkernel.examples.simple_agent

# Team scenario, local backend
LLM_API_KEY=... uv run python -m agentkernel.examples.team_scenario
```

### Run examples on Kubernetes

```bash
# run_k8s_scenario.sh runs the scenario against the configured k8s cluster
LLM_API_KEY=... ./agentkernel/scripts/run_k8s_scenario.sh simple_agent
LLM_API_KEY=... ./agentkernel/scripts/run_k8s_scenario.sh team_scenario
```

### Build and push the base image (k8s only)

```bash
./scripts/build_base_image.sh --force-base
```

### Clean up cluster resources (k8s only)

```bash
./agentkernel/scripts/cleanup_k8s.sh           # delete all agentkernel pods/svc/cm
./agentkernel/scripts/cleanup_k8s.sh --dry-run  # preview what would be deleted
```
