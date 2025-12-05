"""
CodingEnv
---------
Client-side wrapper for the Coding environment server.
Talks HTTP to a single base_url exposing: /reset and /step.

- users instantiate CodingEnv with a base_url provided by the higher-level
  vector/orchestration layer.
- Environment authors ship the Docker image that serves the HTTP API.

(Seeds, episode IDs, request IDs, capabilities can be added later in the payloads.)
"""

from __future__ import annotations

# Support both standalone and in-repo imports
try:
    # Standalone imports (when installed from pip)
    from openenv_core.client_types import StepResult
    from openenv_core.http_env_client import HTTPEnvClient
except ImportError:
    # In-repo imports (when running from OpenEnv repository)
    from core.client_types import StepResult
    from core.http_env_client import HTTPEnvClient

# Use relative imports for sibling modules - works in both modes
from .models import CodeAction, CodeObservation, CodeState


class CodingEnv(HTTPEnvClient[CodeAction, CodeObservation]):
    # --- HTTPEnvClient abstract hooks ---

    def _step_payload(self, action: CodeAction) -> dict:
        # Shape expected by the server's /step endpoint under "action"
        return {
            "code": action.code,
        }

    def _parse_result(self, payload: dict) -> StepResult[CodeObservation]:
        # Expecting: { "observation": {...}, "reward": <float|null>, "done": <bool>, "info": {...} }
        obs = CodeObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> CodeState:
        """
        Parse server response into CodeState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            CodeState object with episode_id, step_count, and last_exit_code
        """
        return CodeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
        )
