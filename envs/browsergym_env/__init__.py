"""BrowserGym Environment for OpenEnv.

BrowserGym is a unified framework for web-based agent tasks that provides
access to multiple benchmarks under a single Gymnasium-compatible API.

Included Benchmarks:
- **MiniWoB++**: 100+ simple web tasks for training (no external infrastructure!)
- **WebArena**: 812 realistic evaluation tasks (requires backend setup)
- **VisualWebArena**: Visual web navigation tasks
- **WorkArena**: Enterprise task automation

Key Features:
- Unified API across all benchmarks
- Gymnasium-compatible interface
- Support for multiple observation types (text, visual, DOM)
- Action spaces for natural language commands
- Perfect for training (MiniWoB) and evaluation (WebArena)

Training Example (MiniWoB - works immediately):
    ```python
    from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

    # Create training environment - no backend setup needed!
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        environment={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "click-test",
        }
    )

    # Train your agent
    for episode in range(1000):
        result = env.reset()
        while not result.done:
            action = agent.get_action(result.observation)
            result = env.step(action)

    env.close()
    ```

Evaluation Example (WebArena - requires backend):
    ```python
    from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

    # Create evaluation environment
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        environment={
            "BROWSERGYM_BENCHMARK": "webarena",
            "BROWSERGYM_TASK_NAME": "0",
            "SHOPPING": "http://your-server:7770",
            # ... other backend URLs
        }
    )

    # Evaluate your trained agent
    result = env.reset()
    # ... run evaluation
    env.close()
    ```
"""

from .client import BrowserGymEnv
from .models import BrowserGymAction, BrowserGymObservation, BrowserGymState

__all__ = [
    "BrowserGymEnv",
    "BrowserGymAction",
    "BrowserGymObservation",
    "BrowserGymState",
]
