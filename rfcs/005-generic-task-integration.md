# RFC: Generic Custom Task Integration

**Status**: Draft
**Created**: 06/12/2025
**Authors**: @atchudhansg
**RFC ID:** 004

## Summary
This RFC proposes a standardized mechanism for injecting custom task logic into OpenEnv environments at runtime. This allows users to define domain-specific tasks, rewards, and termination conditions without modifying the core environment code or rebuilding Docker images.

## Motivation
Currently, adding new tasks to an OpenEnv environment (like BrowserGym) typically requires:
1.  Modifying the source code of the environment.
2.  Rebuilding the Docker image.
3.  Waiting for upstream PRs to be merged for official benchmarks.

This slows down research and prototyping. Users often need to:
-   Test agents on proprietary or local tasks.
-   Rapidly iterate on reward functions.
-   Create custom curricula of tasks.

We need a way to "plug in" task definitions dynamically.

## Proposal

### 1. Task Interface
We define a standard protocol that any custom task must implement. This should be generic enough for various environments (Browser, Terminal, etc.).

```python
class CustomTask:
    def setup(self, config: dict) -> None:
        """Initialize task resources."""
        pass

    def get_observation(self, env_state: Any) -> dict:
        """Transform raw environment state into agent observation."""
        pass

    def calculate_reward(self, state: Any, action: Any, result: Any) -> float:
        """Compute reward for the transition."""
        pass

    def check_done(self, state: Any) -> bool:
        """Determine if the episode should terminate."""
        pass
```

### 2. Injection Mechanism
Environments should support loading these tasks from a specific directory or module path at runtime.

-   **Volume Mount**: Users mount their task code to a standard path (e.g., `/opt/openenv/custom_tasks`).
-   **Dynamic Loading**: The environment server scans this directory and registers valid task classes.
-   **Configuration**: A standard environment variable (e.g., `OPENENV_CUSTOM_TASK_DIR`) tells the server where to look.

### 3. Usage Example (BrowserGym)
In the context of BrowserGym (as implemented in PR #X), this looks like:

```python
# User defines a task locally
class MyTask(CustomTask):
    ...

# User runs the environment with the custom task
env = BrowserGymEnv(
    environment={
        "BROWSERGYM_BENCHMARK": "custom",
        "BROWSERGYM_TASK_NAME": "my-task"
    },
    volumes={
        "/local/path/to/tasks": "/opt/openenv/custom_tasks"
    }
)
```

## Benefits
-   **Decoupling**: Task logic is separate from environment infrastructure.
-   **Speed**: No rebuilds required to change a reward function.
-   **Flexibility**: Supports private/proprietary tasks that cannot be upstreamed.

## Future Work
-   Define schemas for task configuration.
-   Support for remote task definitions (loading from URL/Git).
