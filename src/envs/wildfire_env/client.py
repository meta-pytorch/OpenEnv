# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.http_env_client import HTTPEnvClient
    from core.client_types import StepResult
    from .models import WildfireAction, WildfireObservation, WildfireState
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.http_env_client import HTTPEnvClient
    from openenv_core.client_types import StepResult
    from wildfire_env.models import WildfireAction, WildfireObservation, WildfireState

class WildfireEnv(HTTPEnvClient[WildfireAction, WildfireObservation]):
    def _step_payload(self, action: WildfireAction) -> dict:
        return {"action": action.action, "x": action.x, "y": action.y}

    def _parse_result(self, payload: dict) -> StepResult[WildfireObservation]:
        obs = WildfireObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WildfireState:
        return WildfireState(**payload)


def render_grid(obs: WildfireObservation) -> str:
    legend = {0:"⬛", 1:"🟩", 2:"🟥", 3:"🟫", 4:"🟦"}
    w, h = obs.width, obs.height
    g = obs.grid
    rows = []
    for y in range(h):
        rows.append("".join(legend.get(g[y*w+x], "?") for x in range(w)))
    meta = f"step={obs.step} wind={obs.wind_dir} hum={obs.humidity:.2f} burning={obs.burning_count} burned={obs.burned_count}"
    return "\n".join(rows + [meta])
