# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MazeEnv HTTP Client.

This module provides the client for connecting to a Maze Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import MazeAction, MazeObservation, MazeState

if TYPE_CHECKING:
    pass


class MazeEnv(HTTPEnvClient[MazeAction, MazeObservation]):
    """HTTP client for Maze Environment."""

    def render_ascii_maze(
        self,
        maze: List[List[int]],
        position: List[int],
        start: List[int],
        goal: List[int],
    ) -> None:
        """
        Render the maze grid as ASCII art in the terminal.
        - 0 = free cell
        - 1 = wall
        - S = start
        - G = goal
        - P = player
        - E = exit
        """
        print("\nCurrent Maze State:")
        rows, cols = len(maze), len(maze[0])
        for r in range(rows):
            line = ""
            for c in range(cols):
                if [r, c] == position:
                    line += "P "
                elif [r, c] == start:
                    line += "S "
                elif [r, c] == goal:
                    line += "G "
                elif maze[r][c] == 1:
                    line += "█ "
                elif r == rows - 1 and c == cols - 1:
                    line += "E "
                else:
                    line += ". "
            print(line)
        print()

    def _step_payload(self, action: MazeAction) -> Dict[str, Any]:
        """Prepare payload to send to the environment server."""
        return {"action": action.action}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MazeObservation]:
        """Parse the response from the server into MazeObservation + reward/done."""
        obs_data = payload.get("observation", {})

        observation = MazeObservation(
            position=obs_data.get("position", []),
            total_reward=obs_data.get("total_reward", 0.0),
            legal_actions=obs_data.get("legal_actions", []),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MazeState:
        """Parse environment state from payload."""
        return MazeState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            done=payload.get("done", False),
        )
