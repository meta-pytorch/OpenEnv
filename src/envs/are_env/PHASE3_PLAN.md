# Phase 3 Implementation Plan - Incremental Sub-Phases

## Overview

Phase 3 builds on the completed Phase 1 (skeleton) and Phase 2 (basic ARE integration) by implementing the full functionality in testable increments:

1. **Sub-Phase 3.1**: Initialization & Server Tests
2. **Sub-Phase 3.2**: Tool Calling & Tests
3. **Sub-Phase 3.3**: Ticking Mechanism & Tests

Each sub-phase will include implementation and corresponding tests before moving to the next.

---

## Sub-Phase 3.1: Initialization & Server Tests

### Goals
- Implement real scenario loading in `_handle_initialize()`
- Set up the controlled ARE environment properly
- Add comprehensive tests for server initialization

### Implementation Tasks

#### 1. Update `AREEnvironment.__init__()`
Replace dummy initialization with real ARE environment setup:

```python
from are.simulation.environment import Environment as ARESimulationEnvironment, EnvironmentConfig
from are.simulation.scenarios.scenario import Scenario
from are.simulation.benchmark.local_loader import load_scenario

class AREEnvironment(Environment):
    def __init__(self):
        self._are_env: Optional[ARESimulationEnvironment] = None
        self._scenario: Optional[Scenario] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._notifications_buffer: list[dict] = []
        self._scenario_loaded = False
```

#### 2. Implement `_handle_initialize()`
Load scenarios from JSON and set up ARE environment:

```python
def _handle_initialize(self, action: InitializeAction) -> AREObservation:
    """Load and initialize a scenario"""
    try:
        # Load scenario from path or JSON string
        if action.scenario_path.endswith('.json'):
            with open(action.scenario_path, 'r') as f:
                scenario_json = f.read()
        else:
            scenario_json = action.scenario_path  # Assume it's JSON string

        # Use ARE's scenario loader
        scenario, _ = load_scenario(
            scenario_json,
            scenario_id=f"scenario_{self._state.episode_id}",
            load_completed_events=False
        )

        # Apply config overrides if provided
        if action.scenario_config:
            # Update scenario config here
            pass

        # Initialize scenario
        scenario.initialize()
        self._scenario = scenario

        # Create ARE environment
        config = EnvironmentConfig(
            start_time=0,
            duration=None,  # No duration limit for now
            time_increment_in_seconds=scenario.time_increment_in_seconds,
            oracle_mode=True,  # We control events
            exit_when_no_events=False,
            queue_based_loop=False,  # Time-based
            verbose=False
        )

        self._are_env = ARESimulationEnvironment(config=config)

        # Run scenario but don't start event loop yet
        self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

        # Pause immediately to control ticking
        self._are_env.pause()

        self._scenario_loaded = True

        return self._make_observation(
            action_success=True,
            action_result={"scenario_id": scenario.scenario_id, "duration": scenario.duration},
            environment_state="RUNNING"
        )
    except Exception as e:
        return self._make_observation(
            action_success=False,
            action_error=str(e),
            environment_state="FAILED"
        )
```

#### 3. Update `reset()`
Ensure proper cleanup:

```python
def reset(self) -> AREObservation:
    """Reset clears the ARE environment"""
    if self._are_env is not None:
        self._are_env.stop()
        self._are_env = None

    self._scenario = None
    self._scenario_loaded = False
    self._state = State(episode_id=str(uuid4()), step_count=0)
    self._notifications_buffer = []

    return self._make_observation(
        action_success=True,
        action_result=None,
        environment_state="SETUP"
    )
```

### Testing Tasks

#### Test File: `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_initialization.py`

```python
import pytest
import json
from pathlib import Path
from envs.are_env.server.are_environment import AREEnvironment
from envs.are_env.models import InitializeAction

@pytest.fixture
def simple_scenario_json():
    """Minimal valid scenario JSON"""
    return json.dumps({
        "scenario_id": "test_scenario",
        "duration": 100,
        "time_increment_in_seconds": 1,
        "apps": [],
        "events": []
    })

@pytest.fixture
def are_env():
    """Create fresh environment for each test"""
    env = AREEnvironment()
    yield env
    # Cleanup
    if env._are_env is not None:
        env._are_env.stop()

def test_reset_initializes_clean_state(are_env):
    """Test that reset() creates clean state"""
    obs = are_env.reset()

    assert obs.action_success is True
    assert obs.environment_state == "SETUP"
    assert obs.current_time == 0.0
    assert obs.tick_count == 0
    assert are_env._are_env is None
    assert are_env._scenario is None
    assert are_env._scenario_loaded is False

def test_initialize_with_valid_json_string(are_env, simple_scenario_json):
    """Test initializing with JSON string"""
    are_env.reset()

    action = InitializeAction(scenario_path=simple_scenario_json)
    obs = are_env.step(action)

    assert obs.action_success is True
    assert obs.environment_state == "RUNNING"
    assert are_env._scenario_loaded is True
    assert are_env._are_env is not None
    assert are_env._scenario is not None

def test_initialize_with_file_path(are_env, tmp_path, simple_scenario_json):
    """Test initializing with file path"""
    # Write scenario to temp file
    scenario_file = tmp_path / "test_scenario.json"
    scenario_file.write_text(simple_scenario_json)

    are_env.reset()
    action = InitializeAction(scenario_path=str(scenario_file))
    obs = are_env.step(action)

    assert obs.action_success is True
    assert obs.environment_state == "RUNNING"

def test_initialize_invalid_json(are_env):
    """Test that invalid JSON returns error"""
    are_env.reset()

    action = InitializeAction(scenario_path="invalid json {{{")
    obs = are_env.step(action)

    assert obs.action_success is False
    assert obs.action_error is not None
    assert obs.environment_state == "FAILED"

def test_initialize_missing_required_fields(are_env):
    """Test scenario missing required fields"""
    are_env.reset()

    invalid_scenario = json.dumps({"scenario_id": "test"})  # Missing duration, etc.
    action = InitializeAction(scenario_path=invalid_scenario)
    obs = are_env.step(action)

    assert obs.action_success is False
    assert obs.action_error is not None

def test_reset_after_initialization(are_env, simple_scenario_json):
    """Test that reset cleans up after initialization"""
    are_env.reset()

    # Initialize
    action = InitializeAction(scenario_path=simple_scenario_json)
    are_env.step(action)
    assert are_env._scenario_loaded is True

    # Reset again
    obs = are_env.reset()

    assert obs.action_success is True
    assert obs.environment_state == "SETUP"
    assert are_env._are_env is None
    assert are_env._scenario_loaded is False

def test_state_property_tracking(are_env, simple_scenario_json):
    """Test that state property tracks episode correctly"""
    obs1 = are_env.reset()
    episode_id_1 = are_env.state.episode_id

    # Initialize and step
    action = InitializeAction(scenario_path=simple_scenario_json)
    are_env.step(action)

    assert are_env.state.episode_id == episode_id_1
    assert are_env.state.step_count == 1

    # Reset should change episode ID
    obs2 = are_env.reset()
    episode_id_2 = are_env.state.episode_id

    assert episode_id_2 != episode_id_1
    assert are_env.state.step_count == 0
```

### Acceptance Criteria for Sub-Phase 3.1

- [ ] `AREEnvironment.__init__()` properly sets up internal state
- [ ] `_handle_initialize()` loads scenarios from JSON strings
- [ ] `_handle_initialize()` loads scenarios from file paths
- [ ] `_handle_initialize()` handles invalid JSON gracefully
- [ ] `_handle_initialize()` creates and pauses ARE environment
- [ ] `reset()` properly cleans up ARE environment
- [ ] All initialization tests pass
- [ ] Server can be started and accepts initialization requests via HTTP

---

## Sub-Phase 3.2: Tool Calling & Tests

### Goals
- Implement `_handle_list_apps()` to return real apps/tools
- Implement `_handle_call_tool()` to execute tools
- Handle tool execution errors
- Add comprehensive tests for tool calling

### Implementation Tasks

#### 1. Implement `_handle_list_apps()`
Return actual apps and tools from ARE:

```python
def _handle_list_apps(self, action: ListAppsAction) -> AREObservation:
    """List all available apps and their tools"""
    if not self._scenario_loaded or self._are_env is None:
        return self._make_observation(
            action_success=False,
            action_error="No scenario loaded. Call initialize first.",
            environment_state="SETUP"
        )

    try:
        apps_info = {}

        for app in self._are_env.apps:
            tools = app.get_tools()
            apps_info[app.name] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters  # JSON schema
                }
                for tool in tools
            ]

        return self._make_observation(
            action_success=True,
            action_result={"apps": apps_info},
            available_apps=list(apps_info.keys())
        )
    except Exception as e:
        return self._make_observation(
            action_success=False,
            action_error=str(e)
        )
```

#### 2. Implement `_handle_call_tool()`
Execute tools and optionally advance time:

```python
from are.simulation.events.event import Event, EventType
from are.simulation.events.action import Action

def _handle_call_tool(self, action: CallToolAction) -> AREObservation:
    """Execute a tool on an app"""
    if not self._scenario_loaded or self._are_env is None:
        return self._make_observation(
            action_success=False,
            action_error="No scenario loaded. Call initialize first.",
            environment_state="SETUP"
        )

    try:
        # Get the app
        app = self._are_env.get_app(action.app_name)
        if app is None:
            return self._make_observation(
                action_success=False,
                action_error=f"App '{action.app_name}' not found"
            )

        # Get the tool
        tools = app.get_tools()
        tool = next((t for t in tools if t.name == action.tool_name), None)
        if tool is None:
            return self._make_observation(
                action_success=False,
                action_error=f"Tool '{action.tool_name}' not found on app '{action.app_name}'"
            )

        # Create and execute action
        are_action = Action(
            app=app,
            function=tool.function,
            args=action.tool_args
        )

        # Create event
        event = Event(
            event_id=f"openenv_{uuid4()}",
            event_time=self._are_env.current_time,
            event_type=EventType.AGENT,
            action=are_action
        )

        # Execute event
        completed_event = event.execute()

        # Add to event log
        self._are_env.add_to_log(completed_event)

        # Get result
        tool_result = {
            "success": completed_event.success,
            "result": completed_event.result,
            "error": completed_event.error if not completed_event.success else None
        }

        # Optionally advance time
        if action.advance_time:
            # Resume, tick once, pause
            self._are_env.resume()
            self._are_env.tick()
            self._are_env.pause()

        return self._make_observation(
            action_success=True,
            action_result=tool_result
        )

    except Exception as e:
        return self._make_observation(
            action_success=False,
            action_error=str(e)
        )
```

### Testing Tasks

#### Test File: `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_tool_calling.py`

```python
import pytest
import json
from envs.are_env.server.are_environment import AREEnvironment
from envs.are_env.models import InitializeAction, ListAppsAction, CallToolAction

@pytest.fixture
def scenario_with_apps():
    """Scenario with some basic apps"""
    return json.dumps({
        "scenario_id": "tools_test",
        "duration": 100,
        "time_increment_in_seconds": 1,
        "apps": [
            {
                "app_type": "calendar",
                "app_name": "user_calendar"
            },
            {
                "app_type": "email",
                "app_name": "user_email"
            }
        ],
        "events": []
    })

@pytest.fixture
def initialized_env(scenario_with_apps):
    """Environment with scenario loaded"""
    env = AREEnvironment()
    env.reset()
    env.step(InitializeAction(scenario_path=scenario_with_apps))
    yield env
    if env._are_env is not None:
        env._are_env.stop()

def test_list_apps_without_initialization():
    """Test that list_apps fails without initialization"""
    env = AREEnvironment()
    env.reset()

    obs = env.step(ListAppsAction())

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error

def test_list_apps_returns_available_apps(initialized_env):
    """Test listing apps after initialization"""
    obs = initialized_env.step(ListAppsAction())

    assert obs.action_success is True
    assert "apps" in obs.action_result
    apps = obs.action_result["apps"]

    # Should have calendar and email apps
    assert "user_calendar" in apps or "calendar" in apps
    assert len(apps) > 0

def test_list_apps_includes_tool_info(initialized_env):
    """Test that app listing includes tool details"""
    obs = initialized_env.step(ListAppsAction())

    assert obs.action_success is True
    apps = obs.action_result["apps"]

    # Each app should have tools list
    for app_name, tools in apps.items():
        assert isinstance(tools, list)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

def test_call_tool_without_initialization():
    """Test that tool calling fails without initialization"""
    env = AREEnvironment()
    env.reset()

    obs = env.step(CallToolAction(
        app_name="calendar",
        tool_name="get_events",
        tool_args={}
    ))

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error

def test_call_tool_invalid_app(initialized_env):
    """Test calling tool on non-existent app"""
    obs = initialized_env.step(CallToolAction(
        app_name="nonexistent_app",
        tool_name="some_tool",
        tool_args={}
    ))

    assert obs.action_success is False
    assert "not found" in obs.action_error.lower()

def test_call_tool_invalid_tool(initialized_env):
    """Test calling non-existent tool on valid app"""
    # First get valid app name
    list_obs = initialized_env.step(ListAppsAction())
    app_name = list(list_obs.action_result["apps"].keys())[0]

    obs = initialized_env.step(CallToolAction(
        app_name=app_name,
        tool_name="nonexistent_tool",
        tool_args={}
    ))

    assert obs.action_success is False
    assert "not found" in obs.action_error.lower()

def test_call_tool_with_valid_args(initialized_env):
    """Test successful tool call"""
    # Get available apps and tools
    list_obs = initialized_env.step(ListAppsAction())
    apps = list_obs.action_result["apps"]

    # Find first app with tools
    app_name = None
    tool_name = None
    for name, tools in apps.items():
        if len(tools) > 0:
            app_name = name
            tool_name = tools[0]["name"]
            break

    if app_name and tool_name:
        obs = initialized_env.step(CallToolAction(
            app_name=app_name,
            tool_name=tool_name,
            tool_args={},
            advance_time=False
        ))

        # Tool may or may not succeed depending on args,
        # but the action should be processed
        assert obs.action_success is True
        assert obs.action_result is not None
        assert "result" in obs.action_result or "error" in obs.action_result

def test_call_tool_advances_time_when_requested(initialized_env):
    """Test that advance_time=True increments time"""
    # Get initial time
    initial_time = initialized_env._are_env.current_time
    initial_ticks = initialized_env._are_env.tick_count

    # Get valid app/tool
    list_obs = initialized_env.step(ListAppsAction())
    apps = list_obs.action_result["apps"]
    app_name = list(apps.keys())[0]
    tool_name = apps[app_name][0]["name"]

    # Call tool with advance_time=True
    obs = initialized_env.step(CallToolAction(
        app_name=app_name,
        tool_name=tool_name,
        tool_args={},
        advance_time=True
    ))

    # Time should have advanced
    assert initialized_env._are_env.current_time > initial_time
    assert initialized_env._are_env.tick_count > initial_ticks

def test_call_tool_no_time_advance_when_not_requested(initialized_env):
    """Test that advance_time=False keeps time same"""
    # Get initial time
    initial_time = initialized_env._are_env.current_time
    initial_ticks = initialized_env._are_env.tick_count

    # Get valid app/tool
    list_obs = initialized_env.step(ListAppsAction())
    apps = list_obs.action_result["apps"]
    app_name = list(apps.keys())[0]
    tool_name = apps[app_name][0]["name"]

    # Call tool with advance_time=False
    obs = initialized_env.step(CallToolAction(
        app_name=app_name,
        tool_name=tool_name,
        tool_args={},
        advance_time=False
    ))

    # Time should NOT have advanced
    assert initialized_env._are_env.current_time == initial_time
    assert initialized_env._are_env.tick_count == initial_ticks
```

### Acceptance Criteria for Sub-Phase 3.2

- [ ] `_handle_list_apps()` returns real apps from ARE environment
- [ ] `_handle_list_apps()` includes tool names, descriptions, and parameters
- [ ] `_handle_call_tool()` can execute tools on apps
- [ ] `_handle_call_tool()` handles invalid app names gracefully
- [ ] `_handle_call_tool()` handles invalid tool names gracefully
- [ ] `_handle_call_tool()` respects `advance_time` parameter
- [ ] Tool execution results are captured and returned
- [ ] All tool calling tests pass

---

## Sub-Phase 3.3: Ticking Mechanism & Tests

### Goals
- Implement proper ticking with pause/resume control
- Handle scheduled events that fire during ticks
- Track notifications from events
- Implement `_handle_get_state()` with real data
- Add comprehensive tests for time advancement

### Implementation Tasks

#### 1. Implement `_handle_tick()`
Advance simulation time properly:

```python
def _handle_tick(self, action: TickAction) -> AREObservation:
    """Advance simulation time"""
    if not self._scenario_loaded or self._are_env is None:
        return self._make_observation(
            action_success=False,
            action_error="No scenario loaded. Call initialize first.",
            environment_state="SETUP"
        )

    try:
        # Clear notification buffer before ticking
        self._notifications_buffer.clear()

        # Resume environment
        self._are_env.resume()

        # Execute ticks
        for _ in range(action.num_ticks):
            self._are_env.tick()

        # Pause again
        self._are_env.pause()

        return self._make_observation(
            action_success=True,
            action_result={"ticks_executed": action.num_ticks}
        )

    except Exception as e:
        return self._make_observation(
            action_success=False,
            action_error=str(e)
        )
```

#### 2. Set Up Notification Tracking
Hook into ARE's notification system:

```python
def _setup_notification_handler(self):
    """Set up handler for ARE notifications"""
    def notification_handler(message):
        self._notifications_buffer.append({
            'type': message.message_type.value,
            'message': message.message,
            'timestamp': message.timestamp.isoformat()
        })

    # Register handler
    if self._are_env and hasattr(self._are_env, 'notification_system'):
        self._are_env.notification_system.add_handler(notification_handler)
```

Call this in `_handle_initialize()` after creating ARE environment.

#### 3. Implement `_handle_get_state()`
Return detailed environment state:

```python
def _handle_get_state(self, action: GetStateAction) -> AREObservation:
    """Get detailed environment state"""
    if not self._scenario_loaded or self._are_env is None:
        return self._make_observation(
            action_success=False,
            action_error="No scenario loaded. Call initialize first.",
            environment_state="SETUP"
        )

    try:
        state_info = {}

        if action.include_event_log:
            state_info["event_log"] = [
                {
                    "event_id": e.event_id,
                    "event_time": e.event_time,
                    "event_type": e.event_type.value,
                    "success": e.success
                }
                for e in self._are_env.event_log
            ]

        if action.include_event_queue:
            state_info["event_queue"] = [
                {
                    "event_id": e.event_id,
                    "event_time": e.event_time,
                    "event_type": e.event_type.value
                }
                for e in self._are_env.event_queue
            ]

        if action.include_apps_state:
            state_info["apps_state"] = {
                app.name: app.get_state()
                for app in self._are_env.apps
            }

        return self._make_observation(
            action_success=True,
            action_result=state_info
        )

    except Exception as e:
        return self._make_observation(
            action_success=False,
            action_error=str(e)
        )
```

#### 4. Update `_make_observation()`
Include notifications and current state:

```python
def _make_observation(
    self,
    action_success: bool = True,
    action_result: Optional[dict] = None,
    action_error: Optional[str] = None,
    environment_state: Optional[str] = None
) -> AREObservation:
    """Create observation with current state"""

    if self._are_env:
        current_time = self._are_env.current_time
        tick_count = self._are_env.tick_count
        event_queue_length = len(self._are_env.event_queue)
        event_log_length = len(self._are_env.event_log)
        available_apps = [app.name for app in self._are_env.apps]
        env_state = environment_state or self._are_env.state.value
    else:
        current_time = 0.0
        tick_count = 0
        event_queue_length = 0
        event_log_length = 0
        available_apps = []
        env_state = environment_state or "SETUP"

    # Copy and clear notifications buffer
    notifications = self._notifications_buffer.copy()
    # Don't clear here - let actions control when to clear

    return AREObservation(
        current_time=current_time,
        tick_count=tick_count,
        action_success=action_success,
        action_result=action_result,
        action_error=action_error,
        notifications=notifications,
        environment_state=env_state,
        event_queue_length=event_queue_length,
        event_log_length=event_log_length,
        available_apps=available_apps
    )
```

### Testing Tasks

#### Test File: `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_ticking.py`

```python
import pytest
import json
from envs.are_env.server.are_environment import AREEnvironment
from envs.are_env.models import InitializeAction, TickAction, GetStateAction

@pytest.fixture
def scenario_with_scheduled_events():
    """Scenario with events scheduled at specific times"""
    return json.dumps({
        "scenario_id": "ticking_test",
        "duration": 100,
        "time_increment_in_seconds": 1,
        "apps": [
            {
                "app_type": "calendar",
                "app_name": "test_calendar"
            }
        ],
        "events": [
            {
                "event_id": "event_1",
                "event_time": 5,
                "event_type": "SYSTEM",
                "action": {
                    "app": "test_calendar",
                    "function": "add_event",
                    "args": {"title": "Scheduled Event"}
                }
            },
            {
                "event_id": "event_2",
                "event_time": 10,
                "event_type": "SYSTEM",
                "action": {
                    "app": "test_calendar",
                    "function": "add_event",
                    "args": {"title": "Another Event"}
                }
            }
        ]
    })

@pytest.fixture
def initialized_env(scenario_with_scheduled_events):
    """Environment with scenario loaded"""
    env = AREEnvironment()
    env.reset()
    env.step(InitializeAction(scenario_path=scenario_with_scheduled_events))
    yield env
    if env._are_env is not None:
        env._are_env.stop()

def test_tick_without_initialization():
    """Test that tick fails without initialization"""
    env = AREEnvironment()
    env.reset()

    obs = env.step(TickAction(num_ticks=1))

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error

def test_tick_advances_time(initialized_env):
    """Test that ticking advances simulation time"""
    # Get initial time
    initial_time = initialized_env._are_env.current_time

    # Tick 5 times
    obs = initialized_env.step(TickAction(num_ticks=5))

    assert obs.action_success is True
    assert obs.current_time == initial_time + 5
    assert obs.tick_count == 5

def test_tick_increments_tick_count(initialized_env):
    """Test that tick count increments correctly"""
    # Tick multiple times
    obs1 = initialized_env.step(TickAction(num_ticks=3))
    assert obs1.tick_count == 3

    obs2 = initialized_env.step(TickAction(num_ticks=2))
    assert obs2.tick_count == 5

def test_tick_fires_scheduled_events(initialized_env):
    """Test that scheduled events fire at correct time"""
    # Get state before event
    state_obs = initialized_env.step(GetStateAction(include_event_queue=True))
    initial_queue_length = state_obs.action_result["event_queue"]

    # Tick past first event time (event at time 5)
    obs = initialized_env.step(TickAction(num_ticks=6))

    # Check event log grew
    assert obs.event_log_length > 0

    # Check event was processed
    state_obs = initialized_env.step(GetStateAction(include_event_log=True))
    event_log = state_obs.action_result["event_log"]

    # Should have at least one event
    assert len(event_log) > 0

def test_tick_notifications(initialized_env):
    """Test that notifications are captured during ticks"""
    # Tick and check for notifications
    obs = initialized_env.step(TickAction(num_ticks=10))

    # After ticking past events, should have notifications
    # (depends on ARE's notification system behavior)
    assert isinstance(obs.notifications, list)

def test_get_state_without_initialization():
    """Test that get_state fails without initialization"""
    env = AREEnvironment()
    env.reset()

    obs = env.step(GetStateAction())

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error

def test_get_state_event_log(initialized_env):
    """Test getting event log"""
    # Execute some actions to generate log entries
    initialized_env.step(TickAction(num_ticks=10))

    obs = initialized_env.step(GetStateAction(include_event_log=True))

    assert obs.action_success is True
    assert "event_log" in obs.action_result
    assert isinstance(obs.action_result["event_log"], list)

def test_get_state_event_queue(initialized_env):
    """Test getting event queue"""
    obs = initialized_env.step(GetStateAction(include_event_queue=True))

    assert obs.action_success is True
    assert "event_queue" in obs.action_result
    assert isinstance(obs.action_result["event_queue"], list)

def test_get_state_apps_state(initialized_env):
    """Test getting apps state"""
    obs = initialized_env.step(GetStateAction(include_apps_state=True))

    assert obs.action_success is True
    assert "apps_state" in obs.action_result
    assert isinstance(obs.action_result["apps_state"], dict)

def test_get_state_selective_fields(initialized_env):
    """Test that selective fields work"""
    obs = initialized_env.step(GetStateAction(
        include_event_log=True,
        include_event_queue=False,
        include_apps_state=True
    ))

    assert obs.action_success is True
    assert "event_log" in obs.action_result
    assert "event_queue" not in obs.action_result
    assert "apps_state" in obs.action_result

def test_event_queue_decreases_as_events_fire(initialized_env):
    """Test that event queue shrinks as time advances"""
    # Get initial queue size
    obs1 = initialized_env.step(GetStateAction(include_event_queue=True))
    initial_queue = len(obs1.action_result["event_queue"])

    # Tick past all events
    initialized_env.step(TickAction(num_ticks=20))

    # Get queue size after
    obs2 = initialized_env.step(GetStateAction(include_event_queue=True))
    final_queue = len(obs2.action_result["event_queue"])

    # Queue should be smaller (events moved to log)
    assert final_queue < initial_queue

def test_event_log_grows_as_events_fire(initialized_env):
    """Test that event log grows as time advances"""
    # Get initial log size
    obs1 = initialized_env.step(GetStateAction(include_event_log=True))
    initial_log = len(obs1.action_result["event_log"])

    # Tick past scheduled events
    initialized_env.step(TickAction(num_ticks=10))

    # Get log size after
    obs2 = initialized_env.step(GetStateAction(include_event_log=True))
    final_log = len(obs2.action_result["event_log"])

    # Log should have grown
    assert final_log > initial_log
```

### Acceptance Criteria for Sub-Phase 3.3

- [ ] `_handle_tick()` properly advances time using pause/resume
- [ ] `_handle_tick()` executes correct number of ticks
- [ ] Scheduled events fire at the correct times
- [ ] Notifications are captured and returned in observations
- [ ] `_handle_get_state()` returns event log when requested
- [ ] `_handle_get_state()` returns event queue when requested
- [ ] `_handle_get_state()` returns apps state when requested
- [ ] Event queue shrinks as events are processed
- [ ] Event log grows as events complete
- [ ] All ticking tests pass

---

## Summary of Phase 3 Implementation

### Sub-Phase 3.1: Initialization
**Deliverables:**
- Real scenario loading from JSON
- ARE environment setup with pause/resume control
- Error handling for invalid scenarios
- 7+ initialization tests

### Sub-Phase 3.2: Tool Calling
**Deliverables:**
- List apps with tool details
- Execute tools on apps
- Control time advancement after tool calls
- 8+ tool calling tests

### Sub-Phase 3.3: Ticking
**Deliverables:**
- Time advancement with event firing
- Notification tracking
- Detailed state inspection
- 10+ ticking tests

### Overall Acceptance Criteria

When Phase 3 is complete, the ARE environment should:

✅ Load scenarios from JSON files or strings
✅ Initialize ARE environment with proper control
✅ List all available apps and their tools
✅ Execute tools with configurable time advancement
✅ Advance time through explicit ticking
✅ Fire scheduled events at correct times
✅ Track and return notifications
✅ Provide detailed state information
✅ Handle all error cases gracefully
✅ Pass all 25+ tests

### How to Execute Phase 3

#### Step 1: Sub-Phase 3.1
1. Update `/Users/mortimer/repos/OpenEnv/src/envs/are_env/server/are_environment.py` with initialization code
2. Create `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_initialization.py`
3. Run tests: `pytest tests/envs/test_are_initialization.py -v`
4. Fix any issues until all tests pass
5. Verify HTTP API works with initialization

#### Step 2: Sub-Phase 3.2
1. Update `/Users/mortimer/repos/OpenEnv/src/envs/are_env/server/are_environment.py` with tool calling code
2. Create `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_tool_calling.py`
3. Run tests: `pytest tests/envs/test_are_tool_calling.py -v`
4. Fix any issues until all tests pass
5. Verify HTTP API works with tool calling

#### Step 3: Sub-Phase 3.3
1. Update `/Users/mortimer/repos/OpenEnv/src/envs/are_env/server/are_environment.py` with ticking code
2. Create `/Users/mortimer/repos/OpenEnv/tests/envs/test_are_ticking.py`
3. Run tests: `pytest tests/envs/test_are_ticking.py -v`
4. Fix any issues until all tests pass
5. Verify HTTP API works with ticking

#### Step 4: Integration Testing
1. Run all ARE tests: `pytest tests/envs/test_are_*.py -v`
2. Test full scenario execution through HTTP client
3. Test Docker deployment
4. Update documentation with examples

### Next Steps After Phase 3

Once Phase 3 is complete, consider:
- Phase 4: Observation enrichment (if needed)
- Performance optimization
- Additional scenario examples
- Integration with RL training frameworks
- Web interface enhancements
