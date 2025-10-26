"""Gym-based Connect4 environment wrapped for OpenEnv."""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Tuple
from uuid import uuid4

import numpy as np

from core.env_server.interfaces import Environment

from ..models import Connect4Action, Connect4Observation, Connect4State

# Ensure the third-party Gym env registers itself if present.
try:  # pragma: no cover - optional dependency is best-effort
    importlib.import_module("gym_connect4")
except Exception:  # noqa: BLE001
    pass

try:
    import gym
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The Connect4 environment requires gym>=0.25. "
        "Install it inside your Docker image or development venv."
    ) from exc


def _scalarize_reward(reward: Any) -> float:
    """Map scalar, vector, or ndarray rewards into a single float."""
    if isinstance(reward, (list, tuple, np.ndarray)):
        arr = np.asarray(reward, dtype=float)
        if arr.shape == (2,):
            return float(arr[0] - arr[1])
        return float(arr.sum())
    return float(reward)


def _normalize_board(obs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert arbitrary Connect4 observations into a canonical 6x7 np.ndarray.

    Supports: (obs, info) tuples, 2x6x7 one-hot planes, 6x7x2 one-hot tensors,
    or per-cell vectors embedded in object arrays.
    """
    info: Dict[str, Any] = {}
    board = obs
    if isinstance(obs, tuple) and len(obs) == 2:
        board, info = obs

    arr = np.array(board, dtype=object)

    if arr.ndim == 2 and arr.dtype != object:
        return arr.astype(int), info

    if arr.ndim == 3 and arr.dtype != object and arr.shape[0] == 2:
        return (arr[0].astype(int) - arr[1].astype(int)), info

    if arr.ndim == 3 and arr.dtype != object and arr.shape[2] == 2:
        return (arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)), info

    if (
        arr.ndim == 4
        and arr.dtype != object
        and arr.shape[0] >= 1
        and arr.shape[1] == 3
    ):
        # gym-connect4 returns a list of per-player 3-plane tensors with shape
        # (players, channels=3, width, height). Convert the first player's view
        # (agent perspective) into a signed board matrix.
        player_view = arr[0]  # shape (3, width, height)
        pieces = player_view[1].astype(int) - player_view[2].astype(int)
        # Convert to (rows, cols) with row zero on top.
        return pieces.T, info

    if arr.ndim == 2 and arr.dtype == object:
        h, w = arr.shape
        out = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                val = np.asarray(arr[r, c], dtype=int).ravel()
                if val.size == 2:
                    out[r, c] = int(val[0] - val[1])
                elif val.size == 1:
                    out[r, c] = int(val[0])
        return out, info

    # Fallback: best effort for mismatched shapes
    try:  # pragma: no cover - defensive branch
        if arr.ndim == 3 and arr.shape[0] == 2:
            return (arr[0].astype(int) - arr[1].astype(int)), info
        if arr.ndim == 3 and arr.shape[2] == 2:
            return (arr[:, :, 0].astype(int) - arr[:, :, 1].astype(int)), info
    except Exception:  # noqa: BLE001
        pass

    return np.zeros((6, 7), dtype=int), info


def _legal_actions(board: np.ndarray) -> list[int]:
    return [c for c in range(board.shape[1]) if board[0, c] == 0]


def _current_player(info: Dict[str, Any], board: np.ndarray) -> int:
    try:
        cp = int(info.get("current_player", 0))
        if cp in (1, -1):
            return cp
    except Exception:  # noqa: BLE001
        pass

    p1 = int((board == 1).sum())
    p2 = int((board == -1).sum())
    return 1 if p1 == p2 else -1


class Connect4Environment(Environment):
    """Wrap the gym-connect4 environment so it can be served over HTTP."""

    def __init__(self, gym_id: str | None = None):
        super().__init__()
        self._gym_id = gym_id or os.getenv("GYM_CONNECT4_ID", "Connect4-v0")
        self._env: gym.Env | None = None
        self._state = Connect4State()

    def _ensure_env(self) -> gym.Env:
        if self._env is None:
            self._env = gym.make(self._gym_id)
        return self._env

    def reset(self) -> Connect4Observation:
        env = self._ensure_env()
        raw_obs = env.reset()
        board, info = _normalize_board(raw_obs)
        rows, cols = board.shape
        self._state = Connect4State(
            episode_id=str(uuid4()),
            step_count=0,
            rows=rows,
            cols=cols,
        )

        legal_actions = info.get("legal_actions") if info else None
        if legal_actions is None:
            legal_actions = _legal_actions(board)

        return Connect4Observation(
            board=board.tolist(),
            legal_actions=list(legal_actions),
            current_player=_current_player(info, board),
            last_move=info.get("last_move"),
            reward=0.0,
            done=False,
            info=info,
        )

    def step(self, action: Connect4Action) -> Connect4Observation:  # type: ignore[override]
        env = self._ensure_env()
        result = env.step(int(action.column))

        # Gym 0.25 returns 4-tuple, 0.26+ returns 5-tuple.
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:  # pragma: no cover - defensive branch
            raise RuntimeError(
                f"Unexpected Gym step return type for Connect4: {type(result)}"
            )

        done = bool(terminated or truncated)
        board, info2 = _normalize_board(obs)
        merged_info: Dict[str, Any] = info or {}
        merged_info.update(info2 or {})

        self._state.step_count += 1

        legal_actions = merged_info.get("legal_actions")
        if done:
            legal_actions = []
        elif legal_actions is None:
            legal_actions = _legal_actions(board)

        return Connect4Observation(
            board=board.tolist(),
            legal_actions=list(legal_actions),
            current_player=_current_player(merged_info, board),
            last_move=merged_info.get("last_move"),
            done=done,
            reward=_scalarize_reward(reward),
            info=merged_info,
        )

    @property
    def state(self) -> Connect4State:
        return self._state

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
            self._env = None
