# AgentBus

Communication and storage substrate for agentic infrastructure, enabling safe and fault-tolerant agent workflows.

## Build

```bash
buck2 build //agentbus:agent_bus_server
```

## Run

```bash
# Default port (9999)
buck2 run //agentbus:agent_bus_server

# Custom port
buck2 run //agentbus:agent_bus_server -- --port 8888
AGENT_BUS_PORT=8888 buck2 run //agentbus:agent_bus_server
```

## Test

```bash
# All tests
buck2 test //agentbus:simtests //agentbus:utils_test //agentbus:integration_tests

# Simulator tests (fast, deterministic)
buck2 test //agentbus:simtests

# Integration tests (slower, network-based)
buck2 test //agentbus:integration_tests
```

## API

- `propose(agentBusId, payload)` → Adds command with sequential log position
- `poll(agentBusId, startLogPosition)` → Returns commands from position onwards
