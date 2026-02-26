# AgentKernel

Lightweight, extensible orchestration for teams of LLM-powered agents. Each agent runs as its own process with an independent conversation, tools, and skills. The kernel handles lifecycle, communication, and code shipping — you bring the agent type and the backend.

## Architecture

The core primitive is a **team**: a resource reservation that agents are spawned into. Teams let you allocate capacity once, then spin up and tear down agents within that budget.

AgentKernel factors orchestration into three independent services:

**AgentSpawnerService** — lifecycle. Creates teams, spawns agents into them, kills them. The kernel is agent-type-agnostic: agent-specific config (system prompt, tools, skills) is carried as an opaque `Any` in `spawn_info`, so new agent types can be added without kernel changes. Agents are private to their spawner — `SpawnAgent` returns a cryptographic nonce that must be presented on every subsequent call. This enforces **one agent, one conversation, one owner** without needing sessions or shared state.

**AgentService** — communication. Two planes:
- *Data plane* (`Turn`): send content, stream back the response via SSE. The message body is opaque — the wire format (Anthropic, OpenAI) is set at spawn time via `transport_format`, so the kernel routes without parsing. New protocols can be added by implementing the format on the agent side; the kernel doesn't need to change.
- *Mutation plane* (`Control`): send agent-type-specific operations to a running agent (e.g. hot-load code bundles, list available tools). Payload is again an opaque `Any` keyed by agent type.

**PackagingService** — code shipping. Bundles custom code (helpers, skills, data) into agent images so you don't resend large blobs on every spawn.

The three services are deliberately separated so backends can vary independently: `local` (subprocess, no isolation), `bwrap` (Linux sandbox), `kubernetes` (pods + OCI registry). Adding a new backend means implementing the three protocols — the rest of the kernel, the agent types, and the examples all stay the same.

## Running the demos

**Prerequisites:** Python 3.12+, `LLM_API_KEY` or `OPENAI_API_KEY` env var set.

```bash
# Install
uv sync
```

### Single agent

Spawns one agent, asks a question, streams the response, cleans up.

```bash
python -m agentkernel.examples.simple_agent                            # local (subprocess)
python -m agentkernel.examples.simple_agent --backend bwrap             # bubblewrap sandbox
python -m agentkernel.examples.simple_agent --config agentkernel.yaml   # kubernetes
```

### Team of agents

Spawns an architect + 2 workers with a shared code bundle, has them collaborate, cleans up.

```bash
python -m agentkernel.examples.team_scenario                            # local
python -m agentkernel.examples.team_scenario --backend bwrap             # bubblewrap sandbox
python -m agentkernel.examples.team_scenario --config agentkernel.yaml   # kubernetes
```

### Kubernetes config

Create `agentkernel.yaml` (see `examples/agentkernel.yaml`):

```yaml
backend: kubernetes
namespace: agentkernel-0
base_image: your-registry/agentkernel:latest
kubeconfig: ~/.kube/config
registry_url: your-registry
```

### Verbose logging

```bash
AGENTKERNEL_LOG_LEVEL=DEBUG python -m agentkernel.examples.simple_agent
```

### Tests

```bash
pytest agentkernel/tests/
```
