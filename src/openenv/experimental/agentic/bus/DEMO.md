# AgentBus Demo

Quick guide to running AgentBus with all components.

## Quick Start

### Terminal 1: Start Server
```bash
buck2 run //agentbus:agent_bus_server -- --port 9999
```
Wait for: `Starting AgentBusService Thrift service on port 9999.`

### Terminal 2: Start Decider
```bash
buck2 run //agentbus:decider_service -- \
  --agent-bus-id 1 \
  --policy ON_BY_DEFAULT \
  --host localhost:9999 \
  --poll-interval-ms 500
```
The Decider automatically commits/aborts intentions based on policy.

### Terminal 3: Use CLI (REPL Mode)
```bash
buck2 run //agentbus:agentbus_cli -- \
  --agent-bus-id 1 \
  --host localhost:9999 \
  repl 2>/dev/null
```

**REPL Commands:**
```
agentbus> propose Deploy new feature
agentbus> propose Scale database
agentbus> poll
agentbus> quit
```

## CLI Modes

**REPL (Recommended)** - Interactive shell, builds once:
```bash
buck2 run //agentbus:agentbus_cli -- --agent-bus-id 1 --host localhost:9999 repl 2>/dev/null
```

**One-off commands** - For scripting:
```bash
buck2 run //agentbus:agentbus_cli -- --agent-bus-id 1 --host localhost:9999 propose "Deploy" 2>/dev/null
buck2 run //agentbus:agentbus_cli -- --agent-bus-id 1 --host localhost:9999 poll 2>/dev/null
```

**Tip:** Add `2>/dev/null` to hide buck2 build messages and see only CLI output.

## Environment Variables
```bash
export AGENT_BUS_ID=1
export AGENT_BUS_HOST=localhost:9999

buck2 run //agentbus:agentbus_cli -- repl 2>/dev/null
```

## Voting Policies

**ON_BY_DEFAULT** - Auto-commit intentions:
```bash
--policy ON_BY_DEFAULT
```

**OFF_BY_DEFAULT** - Auto-abort intentions:
```bash
--policy OFF_BY_DEFAULT
```

**FIRST_BOOLEAN_WINS** - Wait for votes:
```bash
--policy FIRST_BOOLEAN_WINS
```

## Multiple Agent Buses

Run separate buses by using different `--agent-bus-id` values. Each can have its own policy:
```bash
# Terminal 2: agent_bus_id=1 with ON_BY_DEFAULT
buck2 run //agentbus:decider_service -- --agent-bus-id 1 --host localhost:9999 --policy ON_BY_DEFAULT

# Terminal 3: agent_bus_id=2 with OFF_BY_DEFAULT
buck2 run //agentbus:decider_service -- --agent-bus-id 2 --host localhost:9999 --policy OFF_BY_DEFAULT
```

## Production

Use `--tiername` instead of `--host`:
```bash
# Decider
buck2 run //agentbus:decider_service -- \
  --agent-bus-id 1 \
  --policy ON_BY_DEFAULT \
  --tiername agentbus.prod

# CLI
buck2 run //agentbus:agentbus_cli -- \
  --agent-bus-id 1 \
  --tiername agentbus.prod \
  propose "Production intention"
```
