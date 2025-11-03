from dataclasses import dataclass
from typing import Any

try:
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required for the web interface. Install template deps with '\n"
        "    pip install -r server/requirements.txt\n'"
    ) from e

from models import Action, Observation, State


class TemplateEnvironment(Environment):
    """Minimal example environment for __ENV_NAME__."""

    def __init__(self) -> None:
        self._state: State = State()

    def reset(self) -> Observation:
        self._state = State(step_count=0)
        return Observation(message="ready")

    def step(self, action: Action) -> Observation:
        # Echo-style placeholder behavior
        payload = getattr(action, "__dict__", {})
        text = str(payload) if payload else ""
        self._state.step_count += 1
        return Observation(message=f"echo: {text}")

    @property
    def state(self) -> State:
        return self._state


env = TemplateEnvironment()
app = create_app(env, Action, Observation, env_name="__ENV_NAME__")


