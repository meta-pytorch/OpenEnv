# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP client for the generic TextArena environment."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.http_env_client import HTTPEnvClient

from .models import (
    TextArenaAction,
    TextArenaMessage,
    TextArenaObservation,
    TextArenaState,
)

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class TextArenaEnv(HTTPEnvClient[TextArenaAction, TextArenaObservation]):
    """HTTP client for the TextArena environment server."""

    def _step_payload(self, action: TextArenaAction) -> Dict[str, Any]:
        return {"message": action.message}

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[TextArenaObservation]:
        obs_data = payload.get("observation", {})
        messages_payload = obs_data.get("messages", [])
        messages = [
            TextArenaMessage(
                sender_id=item.get("sender_id", -1),
                content=item.get("content", ""),
                category=item.get("category", "MESSAGE"),
            )
            for item in messages_payload
            if isinstance(item, dict)
        ]

        observation = TextArenaObservation(
            prompt=obs_data.get("prompt", ""),
            messages=messages,
            current_player_id=obs_data.get("current_player_id", 0),
            legal_players=obs_data.get("legal_players", []),
            info=obs_data.get("info", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TextArenaState:
        return TextArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            env_id=payload.get("env_id", "unknown"),
            num_players=payload.get("num_players", 1),
            max_turns=payload.get("max_turns"),
            turn=payload.get("turn", 0),
            last_reward=payload.get("last_reward", 0.0),
            last_info=payload.get("last_info", {}),
            raw_state=payload.get("raw_state", {}),
        )

