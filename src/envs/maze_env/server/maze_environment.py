# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze Environment Server Implementation.

This module wraps Maze's environment and exposes it
via the OpenEnv Environment interface.
"""

from typing import List, Tuple, Optional
from core.env_server import Environment
from .maze import Maze
from ..models import MazeAction, MazeObservation, MazeState

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "Numpy is not installed. "
        "Please install it following instructions at: "
        "pip install numpy"
    ) from e


class MazeEnvironment(Environment):
    """
    Maze Environment wrapper for OpenEnv.

    This environment wraps Maze game and provides a single-agent interface.

    Args:
        maze_array: Maze array as numpy array
        start cell: Start of the maze
        exit_cell: Exit for the maze
    """

    def __init__(
        self,
        maze_array: np.ndarray,
        start_cell: Tuple[int, int] = (0, 0),
        exit_cell: Optional[Tuple[int, int]] = (7, 7),
    ):
        # Create underlying Maze instance (matches your working code)
        self.env = Maze(maze=maze_array, start_cell=start_cell, exit_cell=exit_cell)
        self.total_reward = 0
        self.start_cell = start_cell
        self.exit_cell = exit_cell
        # env.reset() will be called in reset(); state initialized to None until then
        self.state: Optional[MazeState] = None

    def reset(self) -> MazeObservation:
        """Reset environment and return initial observation (MazeObservation)."""
        observation = (
            self.env.reset()
        )  # typically returns np.array([row, col]) or similar
        # initialize episode state
        self.state = MazeState(episode_id="episode_1", step_count=0, done=False)

        # build MazeObservation; convert numpy to list for JSON-serializable dataclass fields
        pos_list = (
            observation.tolist()
            if hasattr(observation, "tolist")
            else list(observation)
        )
        self.total_reward = 0
        legal_actions = self._compute_legal_actions(pos_list[0])

        return MazeObservation(
            position=pos_list,
            total_reward=self.total_reward,
            legal_actions=legal_actions,
        )

    def step(self, action: MazeAction) -> MazeObservation:
        """
        Step function that manipulates the maze position grid
        and applies rewards/penalties for movement outcomes.
        """

        # --- Get current position ---
        if hasattr(self.env, "agent_position"):
            row, col = self.env.agent_position
        elif hasattr(self.env, "_Maze__current_cell"):
            row, col = self.env._Maze__current_cell
        else:
            row, col = self.env._Maze__start_cell

        maze = np.array(self.env.maze)

        # --- Define movement directions ---
        # 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        move_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        # --- Reward settings ---
        reward_exit = 10.0  # reward for reaching the exit cell
        reward_move = 0.05  # reward for a move that didn't find the exit but is valid
        penalty_visited = -0.25  # penalty for revisiting a cell
        penalty_impossible = -0.75  # penalty for invalid move (wall/outside)

        dr, dc = move_map.get(action.action, (0, 0))
        new_r, new_c = row + dr, col + dc

        # Keep track of visited cells
        if not hasattr(self, "_visited"):
            self._visited = set()
        self._visited.add((row, col))

        # --- Check if move is valid ---
        valid_move = (
            0 <= new_r < maze.shape[0]
            and 0 <= new_c < maze.shape[1]
            and maze[new_r, new_c] != 1
        )

        reward = 0.0
        done = False

        if valid_move:
            # Update position
            row, col = new_r, new_c

            if self.exit_cell and (row, col) == self.exit_cell:
                reward += reward_exit
                done = True
                self._visited = set()
            elif (row, col) in self._visited:
                reward += penalty_visited
            else:
                reward += reward_move
        else:
            # Invalid move
            reward += penalty_impossible

        # --- Update environment position ---
        if hasattr(self.env, "agent_position"):
            self.env.agent_position = (row, col)
        elif hasattr(self.env, "_Maze__current_cell"):
            self.env._Maze__current_cell = (row, col)

        # --- Total reward update ---
        self.total_reward += reward

        # --- Update state ---
        if self.state is None:
            self.state = MazeState(episode_id="episode_1", step_count=0, done=done)
        self.state.step_count += 1
        self.state.done = done

        # --- Observation ---
        pos_list = [row, col]
        legal_actions = self._compute_legal_actions(pos_list)
        # --- Return observation ---
        return MazeObservation(
            position=pos_list,
            total_reward=self.total_reward,
            legal_actions=legal_actions,
            done=done,
        )

    def state(self) -> Optional[MazeState]:
        """Return the current MazeState object."""
        return self.state

    def _compute_legal_actions(self, pos: List[int]) -> List[int]:
        """
        Compute which actions are legal given the current normalized position [row, col].
        (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        actions: List[int] = []
        if not pos or len(pos) < 2:
            return actions

        row, col = int(pos[0]), int(pos[1])
        nrows, ncols = self.env.maze.shape

        # UP
        if row > 0 and self.env.maze[row - 1, col] == 0:
            actions.append(0)
        # DOWN
        if row < nrows - 1 and self.env.maze[row + 1, col] == 0:
            actions.append(1)
        # LEFT
        if col > 0 and self.env.maze[row, col - 1] == 0:
            actions.append(2)
        # RIGHT
        if col < ncols - 1 and self.env.maze[row, col + 1] == 0:
            actions.append(3)

        return actions
