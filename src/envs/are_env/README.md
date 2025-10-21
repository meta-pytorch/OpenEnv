# ARE Environment for OpenEnv

OpenEnv integration for the Agents Research Environment (ARE), which provides event-driven simulations of scenarios involving apps and tool calling.

## Overview

The ARE (Agents Research Environment) is an event-driven simulation framework designed for testing AI agents in realistic scenarios. This OpenEnv wrapper exposes ARE through a simple step/reset API, making it compatible with RL training frameworks.

**Current Status**: Phase 1 - Infrastructure Testing

This is currently a **dummy/echo-style implementation** for testing the HTTP infrastructure. The real ARE integration will be added in Phase 2.

## Installation

### Using Docker (Recommended)

```bash
# Build the base image (from OpenEnv root)
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build the ARE environment image
docker build -f src/envs/are_env/server/Dockerfile -t are-env:latest .
```

### Local Development

```bash
# From OpenEnv root
pip install -e .

# Phase 2: Install ARE (not needed for Phase 1)
# pip install meta-agents-research-environments
```

## Usage

### Python Client

```python
from envs.are_env import AREEnv, InitializeAction, TickAction, CallToolAction

# Start environment via Docker
env = AREEnv.from_docker_image("are-env:latest")

# Reset the environment
result = env.reset()
print(result.observation.environment_state)  # "SETUP"

# Initialize a scenario
action = InitializeAction(scenario_path="/path/to/scenario.json")
result = env.step(action)
print(result.observation.action_success)  # True

# Advance time
result = env.step(TickAction(num_ticks=5))
print(result.observation.current_time)  # 5.0

# Call a tool
action = CallToolAction(
    app_name="calendar",
    tool_name="add_event",
    tool_args={"title": "Meeting", "time": "2pm"},
    advance_time=True
)
result = env.step(action)
print(result.observation.action_result)

# Cleanup
env.close()
```

### Connect to Running Server

```python
from envs.are_env import AREEnv

# Connect to existing server
env = AREEnv(base_url="http://localhost:8000")
result = env.reset()
```

## Action Types

The ARE environment supports multiple action types:

### 1. InitializeAction
Load a scenario into the environment.

```python
from envs.are_env import InitializeAction

action = InitializeAction(
    scenario_path="/path/to/scenario.json",
    scenario_config={"key": "value"}  # Optional overrides
)
```

### 2. TickAction
Advance simulation time.

```python
from envs.are_env import TickAction

action = TickAction(num_ticks=5)  # Advance 5 time steps
```

### 3. ListAppsAction
Get available apps and their tools.

```python
from envs.are_env import ListAppsAction

action = ListAppsAction()
# Returns: {"apps": {"calendar": [...], "email": [...]}}
```

### 4. CallToolAction
Execute a tool on an app.

```python
from envs.are_env import CallToolAction

action = CallToolAction(
    app_name="calendar",
    tool_name="add_event",
    tool_args={"title": "Meeting", "time": "2pm"},
    advance_time=True  # Whether to tick after tool call
)
```

### 5. GetStateAction
Get detailed environment state.

```python
from envs.are_env import GetStateAction

action = GetStateAction(
    include_event_log=True,
    include_event_queue=False,
    include_apps_state=True
)
```

## Observation Structure

Every step returns an `AREObservation` containing:

- `current_time`: Current simulation time
- `tick_count`: Number of ticks elapsed
- `action_success`: Whether the action succeeded
- `action_result`: Result data from the action
- `action_error`: Error message (if any)
- `notifications`: List of events since last step
- `environment_state`: State ("SETUP", "RUNNING", "PAUSED", etc.)
- `event_queue_length`: Number of pending events
- `event_log_length`: Number of completed events
- `available_apps`: List of app names
- `reward`: Numeric reward (1.0 for success, 0.0 for failure)
- `done`: Whether episode is complete

## Development

### Running the Server Locally

```bash
# From OpenEnv root
cd src
uvicorn envs.are_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Phase 1: Basic infrastructure tests
pytest tests/envs/test_are_env.py

# Phase 2+: Full integration tests (coming soon)
```

## Phase 1 vs Phase 2+

**Phase 1** (Current):
- ✅ Directory structure created
- ✅ Action/Observation models defined
- ✅ Dummy server implementation
- ✅ HTTP client implementation
- ✅ Docker setup
- ✅ Basic testing support

**Phase 2** (Coming Next):
- ⏳ Real ARE environment integration
- ⏳ Scenario loading from JSON
- ⏳ Event loop control (pause/resume)
- ⏳ Actual tool execution
- ⏳ Notification system integration

**Phase 3+** (Future):
- ⏳ Full app/tool system
- ⏳ Event log/queue visibility
- ⏳ Comprehensive testing
- ⏳ Example scenarios

## Architecture

The ARE environment uses a **client-server architecture**:

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│   AREEnv        │ ◄──────────────────► │  FastAPI Server │
│  (Client)       │   reset, step, state  │   (Container)   │
└─────────────────┘                       └─────────────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │ AREEnvironment  │
                                          │   (Phase 1:     │
                                          │    Dummy Impl)  │
                                          └─────────────────┘
```

**Phase 2** will replace the dummy implementation with real ARE integration.

## Web Interface

The environment includes a web interface for interactive testing:

```bash
# Start the server
uvicorn envs.are_env.server.app:app --host 0.0.0.0 --port 8000

# Open browser to http://localhost:8000/web
```

The web interface provides:
- Interactive action forms
- Real-time observations
- State inspection
- Action history

## Contributing

See the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed development roadmap.

## License

BSD 3-Clause License (see LICENSE file in repository root)
