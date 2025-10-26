"""Basic smoke tests for the Connect4 OpenEnv environment."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure "src" is on the import path when tests run via pytest.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

pytest.importorskip("gym")

from envs.connect4_env.models import Connect4Action  # noqa: E402
from envs.connect4_env.server.connect4_environment import Connect4Environment  # noqa: E402


def _assert_board_shape(board: list[list[int]], rows: int, cols: int) -> None:
    assert len(board) == rows
    assert all(len(row) == cols for row in board)


def test_connect4_environment_smoke_run() -> None:
    """Reset and step through a short sequence to ensure env wiring works."""
    env = Connect4Environment()

    obs = env.reset()
    _assert_board_shape(obs.board, env.state.rows, env.state.cols)
    assert obs.legal_actions, "Reset should expose at least one legal move"
    assert all(0 <= c < env.state.cols for c in obs.legal_actions)
    assert env.state.step_count == 0

    # Take a handful of legal moves; stop early if the episode terminates.
    max_steps = env.state.rows * 2  # Plenty to detect regressions without full episode
    for expected_step in range(1, max_steps + 1):
        move = obs.legal_actions[0]
        obs = env.step(Connect4Action(column=move))
        assert env.state.step_count == expected_step
        _assert_board_shape(obs.board, env.state.rows, env.state.cols)
        assert isinstance(obs.reward, float)
        if obs.done:
            break
        assert obs.legal_actions, "Episode should offer moves until done"
        assert all(0 <= c < env.state.cols for c in obs.legal_actions)

    env.close()
