# RFC: Generic Task Support

**Status**: Draft
**Created**: 01/24/2026
**Authors**: @atchudhansg
**RFC ID:** 005

## Summary
This RFC proposes a unified interface for defining, registering, and injecting tasks into OpenEnv environments. This allows environments to be decoupled from specific benchmarks or datasets, enabling dynamic task loading for training, evaluation, and custom scenarios.

## Motivation
Currently, tasks are often hardcoded into the environment's `__init__` or tightly coupled with specific benchmark packages (e.g., BrowserGym benchmarks). This limits flexibility:
- Users cannot easily define custom tasks without modifying environment code.
- Switching tasks often requires re-initializing the environment.
- There is no standard way to define "what the agent should do" across different environment types (browser, coding, chat).

As we move towards supporting training workflows and custom evaluations, we need a generic way to tell an environment "Here is a task, reset yourself to this state and evaluate the agent based on these criteria."

### Use Cases
1. **Proprietary Task Evaluation**: Companies need to test agents on internal workflows without upstreaming tasks to public benchmarks
2. **Rapid Prototyping**: Researchers want to iterate on task designs without rebuilding Docker containers
3. **Curriculum Learning**: Training pipelines need to dynamically select tasks based on agent performance
4. **Domain-Specific Tasks**: Custom workflows (e.g., internal tools, enterprise software) that don't fit standard benchmarks

## Architecture Overview

### Component Diagram
```
┌─────────────────────────────────────────────────────────────┐
│  Client Side (Python/TypeScript)                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  env = BrowserGymEnv(...)                            │  │
│  │  env.reset(task="custom/login-test")                 │  │
│  └──────────────────────┬───────────────────────────────┘  │
└─────────────────────────┼──────────────────────────────────┘
                          │ HTTP/REST
                          │
┌─────────────────────────▼──────────────────────────────────┐
│  Docker Container (Server Side)                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Environment Server (FastAPI)                      │   │
│  │  ├─ BrowserGymEnvironment                          │   │
│  │  │  └─ reset(task_id) → loads task from registry   │   │
│  │  └─ CustomBrowserGymEnvironment                    │   │
│  │     ├─ _setup_page(html_content)                   │   │
│  │     ├─ _calculate_reward(page_data, action)        │   │
│  │     └─ _check_done(state)                          │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Task Registry                                     │   │
│  │  ├─ Built-in tasks (copy-paste, login-demo)        │   │
│  │  └─ Mounted tasks (/opt/openenv/custom_tasks/)    │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Task Definitions (Python classes)                 │   │
│  │  - HTML templates                                  │   │
│  │  - Reward functions                                │   │
│  │  - Termination conditions                          │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Client** calls `reset(task="custom/login-test")`
2. **Server** looks up task in registry
3. **Task class** provides HTML template and configuration
4. **Environment** sets up Playwright page with task's HTML
5. **Agent** receives initial observation
6. **Agent** sends actions (e.g., `click('#login-button')`)
7. **Task class** calculates rewards based on page state
8. **Server** returns observation with reward and done flag

## Core Abstractions

### Task Class Interface
All custom tasks must inherit from `CustomBrowserGymEnvironment` and implement:

```python
class CustomBrowserGymEnvironment(ABC):
    task_name: str = "base-task"
    max_steps: int = 10

    @abstractmethod
    def _get_html_content(self) -> str:
        """Return the HTML template for this task."""
        pass

    @abstractmethod
    def _calculate_reward(
        self, page_data: Dict[str, Any], action: str, error: Optional[str] = None
    ) -> float:
        """Calculate reward for the current step."""
        pass

    @abstractmethod
    def _check_done(self, page_data: Dict[str, Any]) -> bool:
        """Determine if the episode should terminate."""
        pass
```

### Task Registry
```python
# Global registry mapping task names to classes
_CUSTOM_TASKS: Dict[str, type] = {}

def register_custom_task(name: str, task_class: type) -> None:
    """Register a custom task."""
    _CUSTOM_TASKS[name] = task_class

def get_custom_task(name: str) -> type:
    """Retrieve a custom task class by name."""
    if name not in _CUSTOM_TASKS:
        raise ValueError(f"Unknown custom task: {name}")
    return _CUSTOM_TASKS[name]
```

### Environment Integration
```python
class BrowserGymEnvironment(Environment):
    def __init__(self, benchmark: str = "miniwob", task_name: str = None, ...):
        if benchmark == "custom":
            # Load custom task
            task_class = get_custom_task(task_name)
            self._custom_env = task_class(...)
        else:
            # Use official BrowserGym benchmark
            self._gym_env = gym.make(f"{benchmark}.{task_name}")
```

## Key Design Decisions

### Decision 1: Python Classes vs. Declarative Config
**Chosen**: Python classes for task definitions

**Rationale**:
- ✅ **Flexibility**: Complex reward logic, multi-step state tracking, dynamic HTML generation
- ✅ **Type Safety**: IDE support, runtime validation via type hints
- ✅ **Debugging**: Standard Python debugging tools work
- ✅ **Reusability**: Tasks can inherit from base classes, share utilities

**Trade-offs**:
- ❌ **Barrier to Entry**: Requires Python knowledge vs. YAML/JSON config
- ❌ **Sandboxing**: Python code has full container privileges
- **Mitigation**: Provide task templates and examples; future work on validation

**Alternative Considered**: YAML/JSON task definitions
```yaml
task:
  name: login-test
  html: tasks/login.html
  reward:
    type: element_present
    selector: "#success-message"
```
- Rejected because complex tasks (multi-step, stateful rewards) require code anyway

### Decision 2: Server-Side Only (No Client-Side Injection)
**Chosen**: Tasks are always server-side within Docker containers

**Rationale**:
- ✅ **Security**: Maintains environment boundary, prevents reward tampering
- ✅ **Consistency**: RFC 002 invariant "rewards inside environment"
- ✅ **Reproducibility**: Task code versioned with container image

**Trade-offs**:
- ❌ **Iteration Speed**: Requires container rebuild or volume mount
- **Mitigation**: Document volume mount workflow for development

### Decision 3: Task Registration via Imports (Not Dynamic Discovery)
**Chosen**: Tasks must be explicitly registered via `@register_custom_task` decorator

**Rationale**:
- ✅ **Predictability**: Clear what tasks are available
- ✅ **Explicit**: No magic filesystem scanning
- ✅ **Control**: Environment chooses which tasks to enable

**Trade-offs**:
- ❌ **Manual Step**: Developers must remember to register tasks
- **Mitigation**: Registration happens automatically via import in `__init__.py`

### Decision 4: Parallel System (Not Replacing BrowserGym Integration)
**Chosen**: Custom tasks are a separate mode (`benchmark="custom"`), not a replacement

**Rationale**:
- ✅ **Backward Compatibility**: Existing code using official benchmarks unaffected
- ✅ **Gradual Adoption**: Users can opt-in to custom tasks
- ✅ **Isolation**: Custom task bugs don't break official benchmark support

**Trade-offs**:
- ❌ **Code Duplication**: Some overlap in action parsing, observation conversion
- **Future Work**: Extract common abstractions to shared base class

## Migration Path

### For Existing BrowserGym Users
No changes required. Continue using official benchmarks:
```python
env = BrowserGymEnv(benchmark="miniwob", task_name="click-test")
```

### For New Custom Task Users
1. **Define Task Class** in `server/custom/custom_tasks.py`
2. **Register Task** via `@register_custom_task("my-task")` decorator
3. **Use Custom Benchmark** mode:
```python
env = BrowserGymEnv(benchmark="custom", task_name="my-task")
```

### Coexistence
Both systems run side-by-side:
```python
# Official benchmark
env1 = BrowserGymEnv(benchmark="miniwob", task_name="click-test")

# Custom task
env2 = BrowserGymEnv(benchmark="custom", task_name="copy-paste")
```

## Implementation Details

### File Organization
```
src/envs/browsergym_env/
├── server/
│   ├── app.py                    # FastAPI server
│   ├── browsergym_environment.py # Main environment class
│   └── custom/
│       ├── __init__.py           # Auto-registers tasks
│       ├── custom_base.py        # Base class for custom tasks
│       ├── custom_models.py      # Data models
│       ├── custom_tasks.py       # Task registry + built-in tasks
│       └── tasks/
│           ├── copy_paste.py     # Example task
│           └── login_demo.py     # Example task
```

### Task Development Workflow

#### Option 1: Build-Time Inclusion (Production)
1. Add task to `server/custom/tasks/my_task.py`
2. Register in `server/custom/__init__.py`
3. Rebuild Docker image
4. Deploy

#### Option 2: Volume Mount (Development)
1. Create task locally: `/local/dev/my_task.py`
2. Mount into container:
```bash
docker run -v /local/dev:/opt/openenv/custom_tasks \
  -e BROWSERGYM_BENCHMARK=custom \
  -e BROWSERGYM_TASK_NAME=my-task \
  browsergym-env:latest
```
3. Environment imports tasks from mounted directory
4. Iterate without rebuilding

### HTML Template Storage
Custom tasks define HTML inline or load from files:

**Option A: Inline HTML**
```python
def _get_html_content(self) -> str:
    return """
    <!DOCTYPE html>
    <html><body>
        <button id="target">Click Me</button>
    </body></html>
    """
```

**Option B: External File**
```python
def _get_html_content(self) -> str:
    template_path = Path(__file__).parent / "templates" / "login.html"
    return template_path.read_text()
```

## Rejection Criteria

When **NOT** to use custom tasks (use official BrowserGym benchmarks instead):

### ❌ Task Belongs in Upstream Benchmark
If the task is:
- Generalizable across domains (not company-specific)
- Well-defined evaluation criteria
- Useful for the broader research community

**Action**: Contribute to BrowserGym's MiniWoB++, WebArena, or WorkArena

### ❌ Task Requires Complex Multi-Page Workflows
Custom tasks are designed for single-page or simple multi-tab scenarios. For complex navigation:
- Multiple domain interactions (e.g., booking.com → airline.com → hotel.com)
- Long chains of dependencies
- State persistence across sessions

**Action**: Use WebArena or WorkArena benchmarks

### ❌ Task Needs Real External Services
Custom tasks use static HTML or local servers. If you need:
- Live API integrations (Stripe, AWS, etc.)
- Real authentication flows
- Production systems

**Action**: Use WorkArena or set up dedicated test environments

### ✅ Good Use Cases for Custom Tasks
- **Internal tool testing**: Company-specific UIs not in public benchmarks
- **Controlled experiments**: A/B testing UI variations for RL research
- **Curriculum learning**: Progressively harder versions of a core task
- **Toy problems**: Simple environments for debugging agent logic

## Proposal

### 1. The Task Abstraction
We define a `Task` as a portable unit of work for an agent. A task definition should ideally be serializable (JSON/YAML) to allow for easy storage and transmission, though some complex tasks may require code.

A `Task` generally consists of:
- **Metadata**: ID, name, description, tags.
- **Instruction**: The prompt or goal given to the agent (e.g., "Book a flight to NYC").
- **Environment Configuration**: Initial state required for the task.
    - *Browser*: Initial URL, cookies, local storage, HTML content.
    - *Coding*: Initial file tree, git repo state.
    - *Game*: Level configuration, seed.
- **Reward/Evaluation Logic**: How to measure success.
    - This can be a reference to a pre-defined reward function (e.g., `reward_function: "exact_match"`).
    - Or a custom script/snippet if the environment supports sandboxed execution of evaluation code.

### 2. Environment Interface Update
We propose updating the `Environment.reset` method in the base `Environment` class to accept a task definition.

```python
class Environment(ABC):
    @abstractmethod
    def reset(self, task: str | dict | None = None, **kwargs) -> Observation:
        """
        Reset the environment.
        
        Args:
            task: Can be:
                  - A string (Task ID) to look up in a registry.
                  - A dictionary/object defining the task configuration directly.
                  - None (default behavior, e.g., random task from loaded benchmark).
        """
        pass
```

### 3. Task Registry
To manage tasks, we introduce a `TaskRegistry`. This allows users to register custom tasks and refer to them by ID.

```python
# Conceptual usage
from openenv.core.tasks import registry

registry.register(
    task_id="custom/login-test",
    env_type="browsergym",
    config={
        "start_url": "http://localhost:8000/login",
        "goal": "Login with user 'admin' and password '1234'",
        "reward_fn": "check_url_contains('dashboard')"
    }
)

# In the environment
env.reset(task="custom/login-test")
```

### 4. Integration Examples

#### BrowserGym (Based on recent PR)
The recent "Custom Task System" for BrowserGym fits this model perfectly.
- **Config**: Defines the HTML/JS or URL for the task.
- **Reset**: The `BrowserGymEnvironment` reads the task config and uses Playwright to set up the page.

#### Coding Environment
- **Config**: A map of filenames to content, or a git commit hash.
- **Reset**: The environment cleans the workspace and writes the specified files.
- **Eval**: Runs a provided test command (e.g., `pytest test_task.py`).

## Architectural Invariants

This proposal adheres to OpenEnv's core architectural principles:

### Server-Side Task Definition and Execution
**Invariant**: Custom tasks must be defined and executed server-side only, within the Docker container environment boundary.

- **Task classes** are instantiated inside the environment server (e.g., `server/custom/custom_tasks.py`)
- **Reward computation** happens exclusively server-side, following RFC 002's "Environment-Computed Rewards" principle
- **Task injection** occurs via Docker volume mounts or build-time inclusion, never via client-side code injection

Example volume mount approach:
```bash
docker run -v /local/tasks:/opt/openenv/custom_tasks \
  -e OPENENV_TASK_MODULE=custom_tasks.my_task \
  browsergym-env:latest
```

The environment server loads the task class from the mounted volume, but all execution stays within the container.

### Action Execution Model
**Invariant**: Custom tasks inherit the parent environment's action execution model and security boundaries.

For BrowserGym environments:
- Custom tasks use the same Playwright-based action execution as official benchmarks
- JavaScript execution via `page.evaluate()` is inherited from BrowserGym's standard behavior
- This is **not** a violation of the "Dual API boundary" - agents receive actions as strings and have no direct access to `reset()`, `step()`, or `state()` methods
- Browser-level sandboxing (Same-Origin Policy, CSP) is BrowserGym's responsibility

For other environment types:
- Coding environments: Actions execute in sandboxed shell/interpreter
- Game environments: Actions map to discrete game moves
- Chat environments: Actions are text messages with no simulation control

### Dual API Boundary Compliance
**Critical**: Custom tasks must not expose simulation control to agents.

Prohibited:
- ❌ Exposing `reset()`, `step()`, or `state()` via MCP tools or action strings
- ❌ Allowing agents to modify reward computation logic at runtime
- ❌ Providing agents with direct access to the environment's HTTP API

Allowed:
- ✅ Domain-specific actions (browser clicks, shell commands, game moves)
- ✅ Observation data that reflects task state
- ✅ Standard BrowserGym/Playwright actions that manipulate the task environment

## Implementation Plan
1.  Define the `Task` schema/interface in `core`.
2.  Update `Environment.reset` signature (backward compatible).
3.  Implement a basic `TaskRegistry` with server-side task loading.
4.  Document volume mount and build-time task injection patterns.
5.  Refactor `BrowserGymEnvironment` to support the new `reset(task=...)` pattern, leveraging the custom task system logic.
6.  Extend to other environments (Coding, TextArena) incrementally.

## Security Considerations

### Task Isolation
- Custom tasks run in the same Docker container as the environment server
- Task code has the same privileges as the environment (by design)
- Users mounting custom tasks should trust the task code (similar to mounting any code into a container)

### Reward Integrity
- Reward functions are part of the task class definition (server-side)
- Agents cannot modify reward logic via actions
- Reward computation uses only server-side state (page DOM, file system, etc.)

### Future Enhancements
- Sandboxed task validation before loading
- Task signature verification for shared task repositories
- Rate limiting for resource-intensive custom tasks
