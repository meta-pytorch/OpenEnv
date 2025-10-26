"""HTTP client for the Connect4 OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import Connect4Action, Connect4Observation, Connect4State


class Connect4Env(HTTPEnvClient[Connect4Action, Connect4Observation]):
    """Thin HTTP client used by agents to interact with the Connect4 server."""

    def _step_payload(self, action: Connect4Action) -> Dict[str, Any]:
        return {"column": action.column, "metadata": action.metadata}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Connect4Observation]:
        obs_data = payload.get("observation", {})
        observation = Connect4Observation(
            board=obs_data.get("board", []),
            legal_actions=obs_data.get("legal_actions", []),
            current_player=obs_data.get("current_player", 1),
            last_move=obs_data.get("last_move"),
            info=obs_data.get("info", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Connect4State:
        return Connect4State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            rows=payload.get("rows", 6),
            cols=payload.get("cols", 7),
        )
