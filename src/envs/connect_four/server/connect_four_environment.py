from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pyspiel  # OpenSpiel
except Exception as e:
    raise ImportError(
        "open_spiel (pyspiel) is required. Install with `pip install open_spiel`."
    ) from e


@dataclass
class ConnectFourConfig:
    game_string: str = "connect_four"
    # If True, the env auto-plays the opponent (player 1) using a trivial policy
    # whenever it becomes their turn (keeps a single-agent loop simple).
    autoplay_opponent: bool = False
    # Opponent policy: "random" | "lowest" | "highest"
    opponent_policy: str = "random"


class ConnectFourEnvironment:
    """OpenSpiel-backed Connect Four with OpenEnv-compatible semantics."""

    ROWS = 6
    COLS = 7

    def __init__(self, config: Optional[ConnectFourConfig] = None):
        self.config = config or ConnectFourConfig()
        self._game = pyspiel.load_game(self.config.game_string)
        self._state = self._game.new_initial_state()

        # Agent = player 0; opponent = player 1
        self._agent_player: int = 0
        self._move_count: int = 0
        self._episode_id: str = ""
        # cache of reconstructed grid (-1 empty, {0,1} owners)
        self._grid_cache: Optional[np.ndarray] = None

    # ----------------------------- API -----------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self._game.new_initial_state()
        self._move_count = 0
        self._episode_id = self._new_episode_id()
        self._grid_cache = None
        obs = self._build_observation(done=False, reward=0.0, info={"engine": "open_spiel"})
        return obs, self._build_state()

    def step(self, column: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply agent move (column 0..6). Optionally autoplay opponent move."""
        assert 0 <= column < self.COLS, f"column out of range: {column}"

        self._maybe_autoplay_until_agent_turn()

        # Map to OpenSpiel action; legality guard
        act = self._column_to_action(column)
        legal = self._state.legal_actions()
        if act not in legal:
            info = {"error": "illegal_action", "legal_columns": self.legal_actions()}
            obs = self._build_observation(done=True, reward=-1.0, info=info)
            return obs, self._build_state()

        self._state.apply_action(act)
        self._move_count += 1
        self._invalidate_grid_cache()

        if self._state.is_terminal():
            reward = self._terminal_reward_for_agent()
            obs = self._build_observation(done=True, reward=reward, info={"engine": "open_spiel"})
            return obs, self._build_state()

        if self.config.autoplay_opponent:
            self._autoplay_opponent_once()
            if self._state.is_terminal():
                reward = self._terminal_reward_for_agent()
                obs = self._build_observation(done=True, reward=reward, info={"engine": "open_spiel"})
                return obs, self._build_state()

        obs = self._build_observation(done=False, reward=0.0, info={"engine": "open_spiel"})
        return obs, self._build_state()

    def close(self) -> None:
        # No special cleanup required
        self._state = self._game.new_initial_state()
        self._grid_cache = None

    # --------------------------- helpers ---------------------------

    def legal_actions(self) -> List[int]:
        return sorted({self._action_to_column(a) for a in self._state.legal_actions()})

    def current_player(self) -> int:
        return 1 if self._state.current_player() == self._agent_player else -1

    def board_agent_view(self) -> np.ndarray:
        """Return 6x7 board: 0 empty, +1 agent discs, -1 opponent discs."""
        grid = self._reconstruct_grid_from_history()
        board = np.zeros_like(grid, dtype=int)
        board[grid == -1] = 0
        board[grid == self._agent_player] = 1
        board[(grid != -1) & (grid != self._agent_player)] = -1
        return board

    def _reconstruct_grid_from_history(self) -> np.ndarray:
        """Rebuild grid (-1 empty, 0/1 owners) from action history."""
        if self._grid_cache is not None:
            return self._grid_cache
        grid = np.zeros((self.ROWS, self.COLS), dtype=int) - 1  # -1 empty
        player = 0  # starts with player 0
        for act in self._state.history():
            col = self._action_to_column(act)
            rr = self._lowest_empty_row(grid, col)
            if rr is not None:
                grid[rr, col] = player
            player = 1 - player
        self._grid_cache = grid
        return grid

    @staticmethod
    def _lowest_empty_row(grid: np.ndarray, col: int) -> Optional[int]:
        for r in range(grid.shape[0] - 1, -1, -1):
            if grid[r, col] == -1:
                return r
        return None

    def _invalidate_grid_cache(self) -> None:
        self._grid_cache = None

    # ----- action mapping -----

    def _column_to_action(self, col: int) -> int:
        # OpenSpiel uses 0..6 column IDs as actions
        # still verify against legal action list in case of variant configs
        for a in self._state.legal_actions():
            if self._action_to_column(a) == col:
                return a
        return col

    @staticmethod
    def _action_to_column(action: int) -> int:
        return int(action)

    # ----- opponent autoplay -----

    def _maybe_autoplay_until_agent_turn(self) -> None:
        if not self.config.autoplay_opponent:
            return
        while self._state.current_player() != self._agent_player and not self._state.is_terminal():
            self._autoplay_opponent_once()

    def _autoplay_opponent_once(self) -> None:
        if self._state.current_player() == self._agent_player or self._state.is_terminal():
            return
        legal = self._state.legal_actions()
        if not legal:
            return
        cols = [self._action_to_column(a) for a in legal]
        if self.config.opponent_policy == "lowest":
            chosen_col = min(cols)
        elif self.config.opponent_policy == "highest":
            chosen_col = max(cols)
        else:
            chosen_col = int(np.random.choice(cols))
        self._state.apply_action(self._column_to_action(chosen_col))
        self._invalidate_grid_cache()

    # ----- rewards -----

    def _terminal_reward_for_agent(self) -> float:
        if not self._state.is_terminal():
            return 0.0
        returns = self._state.returns()
        val = float(returns[self._agent_player])  # >0 win, <0 loss, 0 draw
        if val > 0:
            return 1.0
        if val < 0:
            return -1.0
        return 0.0

    # ----- payloads -----

    def _build_observation(self, done: bool, reward: float, info: Dict[str, Any]) -> Dict[str, Any]:
        board = self.board_agent_view()
        obs = {
            "board": board.tolist(),
            "legal_actions": [] if done else self.legal_actions(),
            "current_player": self.current_player() if not done else 1,
            "last_move": self._last_move_column(),
            "done": bool(done),
            "reward": float(reward),
            "info": dict(info or {}),
        }
        return obs

    def _build_state(self) -> Dict[str, Any]:
        return {
            "rows": self.ROWS,
            "cols": self.COLS,
            "move_count": self._move_count,
            "episode_id": self._episode_id,
        }

    def _last_move_column(self) -> Optional[int]:
        hist = self._state.history()
        if not hist:
            return None
        return self._action_to_column(hist[-1])

    @staticmethod
    def _new_episode_id() -> str:
        import uuid
        return str(uuid.uuid4())
