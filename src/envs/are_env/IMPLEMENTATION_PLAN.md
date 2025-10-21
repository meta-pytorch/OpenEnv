# ARE Environment - OpenEnv Wrapper Implementation Plan

## Overview

This document outlines the plan to wrap the ARE (Agents Research Environment) within the OpenEnv framework. The ARE environment is event-driven and simulates scenarios involving apps and tool calling. The goal is to expose it through OpenEnv's step/reset/state API.

## Key Challenges

1. **Event-driven vs Step-driven**: ARE uses an automatic event loop with time-based ticking, while OpenEnv uses manual step() calls
2. **Time Management**: ARE's time manager needs to be controlled externally rather than running autonomously
3. **Scenario Loading**: ARE scenarios need to be loaded dynamically through the environment
4. **Action Space**: ARE doesn't have a traditional action space - it's tool-based with dynamic app registration

## Architecture

### ARE Environment Characteristics (from `/Users/mortimer/repos/meta-agents-research-environments/are/simulation/environment.py`)

- **Event Loop**: Can operate in time-based or queue-based mode
- **Time Management**: `TimeManager` controls simulation time with `tick()` method
- **Apps**: Dynamic set of apps, each providing tools
- **Notification System**: Tracks events and messages
- **Scenarios**: Loaded via `run(scenario)` method
- **State**: Tracks event log, event queue, apps state, and time

### OpenEnv Requirements

- **reset()**: Initialize environment and return initial observation
- **step(action)**: Execute action and return observation
- **state**: Property returning episode metadata

## Implementation Plan

### Phase 1: Project Structure Setup

Create the standard OpenEnv directory structure:

```
src/envs/are_env/
├── __init__.py           # Export AREAction, AREObservation, AREEnv
├── models.py             # Define Action, Observation, State dataclasses
├── client.py             # Implement AREEnv(HTTPEnvClient)
├── README.md             # Documentation
└── server/
    ├── __init__.py
    ├── are_environment.py   # Implement AREEnvironment(Environment)
    ├── app.py               # FastAPI app
    └── Dockerfile           # Container image
```

### Phase 2: Define Action and Observation Models

**Action Types** (Union type):

1. **InitializeAction**
   - `action_type: Literal["initialize"]`
   - `scenario_path: str` - Path to scenario JSON/YAML
   - `scenario_config: Optional[dict]` - Override scenario config
   - Purpose: Load a scenario and initialize the ARE environment

2. **TickAction**
   - `action_type: Literal["tick"]`
   - `num_ticks: int = 1` - Number of ticks to advance (default 1)
   - Purpose: Advance simulation time

3. **ListAppsAction**
   - `action_type: Literal["list_apps"]`
   - Purpose: Get available apps and their tools

4. **CallToolAction**
   - `action_type: Literal["call_tool"]`
   - `app_name: str` - Name of the app
   - `tool_name: str` - Name of the tool/function
   - `tool_args: dict[str, Any]` - Arguments for the tool
   - `advance_time: bool = True` - Whether to tick after tool call (default: True)
   - Purpose: Call a tool on a specific app, optionally advancing time

5. **GetStateAction**
   - `action_type: Literal["get_state"]`
   - `include_event_log: bool = True`
   - `include_event_queue: bool = False`
   - `include_apps_state: bool = True`
   - Purpose: Get detailed environment state

**Observation Model**:

```python
@dataclass
class AREObservation(Observation):
    current_time: float  # Current simulation time
    tick_count: int  # Number of ticks elapsed

    # Result of the action
    action_success: bool
    action_result: Optional[dict]  # Tool call result, app list, etc.
    action_error: Optional[str]  # Error message if action failed

    # Notifications since last observation
    notifications: list[dict]  # List of notification events

    # Environment state
    environment_state: str  # SETUP, RUNNING, PAUSED, STOPPED, FAILED
    event_queue_length: int
    event_log_length: int

    # Apps info (lightweight)
    available_apps: Optional[list[str]]  # List of app names

    # Additional metadata
    metadata: dict = field(default_factory=dict)
```

### Phase 3: Server Implementation - AREEnvironment

**Key Design Decisions**:

1. **Subclass ARE Environment**: Inherit from `are.simulation.environment.Environment` for full control
   - Benefits: Direct access to time_manager, event_queue, event_log
   - Benefits: Override threading behavior without pause/resume dance
   - Benefits: Add custom methods for OpenEnv integration
   - Benefits: Full control over tick timing and execution
2. **Override Event Loop**: Disable automatic threading, provide manual tick control
3. **Scenario Management**: Load scenarios via initialize action, reset clears scenario

**Implementation Strategy - Option A: Subclass ARE Environment (Recommended)**:

```python
from are.simulation.environment import Environment as ARESimulationEnvironment

class ControlledAREEnvironment(ARESimulationEnvironment):
    """
    Subclass of ARE Environment with manual control over event loop.
    Disables automatic threading and provides explicit tick() control.
    """

    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        # Disable automatic thread start
        self._manual_control = True

    def start(self, debug: bool = False):
        """Override to prevent automatic thread start"""
        if debug:
            logger.setLevel(logging.DEBUG)

        if self.state == EnvironmentState.RUNNING:
            self.log_debug("Environment already running.")
            return

        self.state = EnvironmentState.RUNNING
        self.prepare_events_for_start()
        # Don't start thread - we'll call tick() manually

    def manual_tick(self, num_ticks: int = 1):
        """Execute N ticks synchronously without threading"""
        for _ in range(num_ticks):
            self.tick()
            self.tick_count += 1

    def run(self, scenario: Scenario, wait_for_end: bool = False, schedule_events: bool = True):
        """Override to prevent automatic event loop"""
        # Same initialization as parent
        if not scenario._initialized:
            raise Exception("Scenario not initialized, call scenario.initialize() first")

        self.log_info(f"Running scenario {scenario.scenario_id} (duration={scenario.duration})")
        self.time_manager.reset(start_time=self.start_time)
        self.duration = scenario.duration
        self.time_increment_in_seconds = scenario.time_increment_in_seconds
        self.delete_all_completed_events()
        self.register_apps(scenario.apps if scenario.apps else [])

        if schedule_events:
            self.schedule(scenario.events)

        # Start but don't wait (no thread created)
        self.start()
        # Explicitly don't call join() - we control ticking


class AREEnvironment(Environment):  # OpenEnv Environment
    def __init__(self):
        self._are_env: Optional[ControlledAREEnvironment] = None
        self._scenario: Optional[Scenario] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._notifications_buffer: list[dict] = []

    def reset(self) -> AREObservation:
        """Reset clears the ARE environment"""
        if self._are_env is not None:
            self._are_env.stop()
            self._are_env = None
        self._scenario = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._notifications_buffer = []
        return self._make_observation(...)

    def step(self, action: AREAction) -> AREObservation:
        """Execute action based on type"""
        if action.action_type == "initialize":
            return self._handle_initialize(action)
        elif action.action_type == "tick":
            return self._handle_tick(action)
        elif action.action_type == "list_apps":
            return self._handle_list_apps(action)
        elif action.action_type == "call_tool":
            return self._handle_call_tool(action)
        elif action.action_type == "get_state":
            return self._handle_get_state(action)
```

**Implementation Strategy - Option B: Composition (Fallback)**:

If subclassing proves difficult, use composition with pause/resume:

```python
class AREEnvironment(Environment):  # OpenEnv Environment
    def __init__(self):
        self._are_env: Optional[ARESimulationEnvironment] = None
        # ... rest stays the same but use pause/resume pattern
```

**Critical ARE Integration Points**:

1. **Initialization**:
   ```python
   # Create ARE environment without auto-start
   config = EnvironmentConfig(
       start_time=0,
       duration=None,  # No duration limit
       time_increment_in_seconds=1,
       oracle_mode=True,  # We control the events
       exit_when_no_events=False,  # Don't auto-exit
       queue_based_loop=False,  # Time-based
       verbose=False
   )
   self._are_env = ARESimulationEnvironment(config=config)

   # Load scenario but don't start event loop
   self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

   # Immediately pause so we control ticking
   self._are_env.pause()
   ```

2. **Ticking**:
   ```python
   # Resume, tick, then pause again
   self._are_env.resume()
   for _ in range(num_ticks):
       self._are_env.tick()
   self._are_env.pause()
   ```

3. **Tool Calling**:
   ```python
   # Get app and call tool directly
   app = self._are_env.get_app(app_name)
   tool = next(t for t in app.get_tools() if t.name == tool_name)

   # Create an event and execute it
   action = Action(app=app, function=tool.function, args=tool_args)
   event = Event(
       event_id=f"openenv_{uuid4()}",
       event_time=self._are_env.current_time,
       event_type=EventType.AGENT,
       action=action
   )
   completed_event = event.execute()
   self._are_env.add_to_log(completed_event)
   ```

4. **Notification Tracking**:
   ```python
   # Hook into notification system to buffer notifications
   def notification_handler(message: Message):
       self._notifications_buffer.append({
           'type': message.message_type.value,
           'message': message.message,
           'timestamp': message.timestamp.isoformat()
       })

   # Register handler on notification system
   self._are_env.notification_system.add_handler(notification_handler)
   ```

### Phase 4: Client Implementation

Standard OpenEnv client pattern:

```python
class AREEnv(HTTPEnvClient[AREAction, AREObservation]):
    """Client for ARE Environment"""

    def __init__(self, base_url: str):
        super().__init__(
            base_url=base_url,
            action_cls=AREAction,
            observation_cls=AREObservation
        )

    @classmethod
    def from_docker_image(cls, image_name: str, ...):
        # Standard docker startup
        ...
```

### Phase 5: Docker Setup

**Dockerfile**:

```dockerfile
FROM python:3.11-slim

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ARE package from pip
RUN pip install meta-agents-research-environments

# Copy OpenEnv core
COPY core/ /app/core/

# Copy ARE env server
COPY src/envs/are_env/server/ /app/

# Install any additional requirements
COPY src/envs/are_env/server/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Note**: ARE is available as `meta-agents-research-environments` on pip, so we can install it directly in the Docker image.

### Phase 6: Scenario Management

**Scenario Loading from JSON**:

Focus on JSON-based scenario loading. ARE provides two main loaders:

1. **HuggingFace Loader** (`are.simulation.benchmark.huggingface_loader`):
   - Loads scenarios from HuggingFace datasets
   - Example: `load_dataset(dataset_name, name=config, streaming=True)`
   - Use `load_scenario(scenario_data, scenario_id, load_completed_events)`

2. **Local Loader** (`are.simulation.benchmark.local_loader`):
   - Loads scenarios from local JSON/JSONL files
   - Use `find_scenario_paths(dataset_path, config, split)`
   - Then `load_scenario(scenario_str, scenario_path, load_completed_events)`

**Implementation Approach**:

```python
from are.simulation.benchmark.scenario_loader import load_scenario

def _load_scenario_from_json(self, scenario_json: str, scenario_id: str):
    """Load a scenario from JSON string"""
    scenario, completed_events = load_scenario(
        scenario_json,
        scenario_id,
        load_completed_events=False  # Don't need oracle events
    )
    return scenario
```

**Scenarios can be**:
- Loaded from file path on server (passed in initialize action)
- Embedded as JSON string in initialize action
- Pre-packaged in Docker image at known locations

### Phase 7: Testing Strategy

**Unit Tests**:
- Test each action type independently
- Mock ARE environment for faster tests
- Test error handling

**Integration Tests**:
- Full scenario execution
- Tool calling sequences
- Time advancement
- State consistency

**Docker Tests**:
- Container startup
- HTTP API
- Full scenario through Docker

## Critical Questions & Decisions

### Q1: How to handle ARE's automatic event loop?

**Decision**:
- Use `oracle_mode=True` to have full control
- Start environment with `run(scenario, wait_for_end=False)`
- Immediately `pause()` after initialization
- On each `tick` action: `resume()` → `tick()` → `pause()`

### Q2: Should tool calls trigger time advancement?

**Options**:
A. Tool calls are instant (no time advancement)
B. Tool calls advance time by 1 tick
C. Tool calls can specify time advancement

**Decision**: Option B with Option A as parameter - tool calls should trigger a tick by default and return the tool results + any new events/notifications from the environment resulting from that tick. Add an optional parameter `advance_time: bool = True` to allow instant tool calls when needed.

### Q3: How to represent ARE's app/tool structure in OpenEnv?

**Decision**:
- `list_apps` returns: `{app_name: [list of tool info]}`
- Tool info from `AppTool` dataclass (from `/Users/mortimer/repos/meta-agents-research-environments/are/simulation/apps/app.py`)
- `AppTool` is serializable and includes: name, description, and parameters schema
- `call_tool` takes app_name + tool_name + args
- Use `app.get_tools()` to retrieve tools from each app

### Q4: How to handle ARE's notification system?

**Decision**:
- Buffer notifications between steps
- Return buffered notifications in observation
- Clear buffer after returning observation
- Agent can use notifications to understand what happened

### Q5: What to do about events scheduled in the scenario?

**Options**:
A. Let events fire when their time comes during ticks
B. Disable all scenario events, only allow tool calls
C. Hybrid: some events auto-fire, agent can also call tools

**Recommendation**: Option A - respect scenario events. When ticking, events at that time will fire. This preserves ARE's event-driven nature.

### Q6: How to handle scenario reset vs environment reset?

**Decision**:
- `reset()` clears everything (scenario, ARE env, state)
- To run a new scenario: `reset()` → `step(initialize(new_scenario))`
- To restart same scenario: need to support re-initialization

## Implementation Phases

### Phase 1: Minimal Skeleton (Start Here)
- [ ] Create directory structure
- [ ] Define basic Action/Observation models (just initialize + tick)
- [ ] Implement dummy server (echo-style, no real ARE integration)
- [ ] Create Docker setup
- [ ] Test basic HTTP communication

### Phase 2: ARE Integration
- [ ] Integrate real ARE Environment
- [ ] Implement initialize action (load scenario)
- [ ] Implement tick action (advance time)
- [ ] Handle ARE's event loop control (pause/resume pattern)
- [ ] Test scenario loading and execution

### Phase 3: Tool Calling
- [ ] Implement list_apps action
- [ ] Implement call_tool action
- [ ] Hook into ARE's app/tool system
- [ ] Test tool execution

### Phase 4: Observation Enrichment
- [ ] Add notification tracking
- [ ] Add event log/queue visibility
- [ ] Add apps state visibility
- [ ] Test observation completeness

### Phase 5: Testing & Documentation
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Create example scenarios
- [ ] Write comprehensive README
- [ ] Test with real RL training loop

## Dependencies

**ARE Environment**:
- Need access to `are.simulation.environment.Environment`
- Need access to `are.simulation.scenarios.scenario.Scenario`
- Need access to `are.simulation.apps.app.App`
- Need access to notification system

**OpenEnv Core**:
- `core.env_server.interfaces.Environment`
- `core.env_server.types.Action, Observation, State`
- `core.env_client.HTTPEnvClient`

## Next Steps

1. **Create Phase 1 skeleton** - Get basic structure working
2. **Test with dummy implementation** - Verify HTTP communication works
3. **Add ARE integration** - Start with simplest possible scenario
4. **Iterate on action space** - Refine based on actual usage

## Open Questions

1. Should we expose ARE's validation system through OpenEnv?
2. How to handle ARE's replay functionality?
3. Should we support multiple scenarios in one environment instance?
4. How to handle long-running scenarios (duration management)?
5. Should we expose ARE's GraphQL cache?
6. How to handle ARE's different environment types (GUI vs CLI)?

## Success Criteria

The ARE OpenEnv wrapper is successful when:
- ✅ Can load and execute ARE scenarios through step/reset API
- ✅ Can control time advancement explicitly
- ✅ Can list and call tools on apps
- ✅ Receives observations with relevant state information
- ✅ Can run in Docker container
- ✅ Can integrate with RL training frameworks

## Notes

- ARE's `queue_based_loop` might be useful but requires `oracle_mode=True`
- Consider exposing ARE's pause/resume directly for debugging
- The notification system is rich - we should surface it well
- ARE's event system is powerful - preserve its capabilities
- Think about how to debug: expose event log, allow introspection
