# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test suite for Minesweeper Environment."""

import sys
import os
from pathlib import Path

# Add src to PYTHONPATH for proper imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_PATH)
os.environ["PYTHONPATH"] = SRC_PATH

from envs.minesweeper_env import (
    MinesweeperAction,
    MinesweeperObservation,
    GameStatus,
    MinesweeperEnv,
)
import subprocess
import unittest
import time
import requests
import signal


class TestMinesweeperEnv(unittest.TestCase):
    """Test cases for the Minesweeper environment."""

    server_process = None

    @classmethod
    def setUpClass(cls):
        """Start the server once for all tests."""
        cls.server_process = subprocess.Popen(
            ["python", "-m", "envs.minesweeper_env.server.app"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)  # Give server time to start

        # Verify server is running
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code != 200:
                raise RuntimeError("Server health check failed")
        except requests.ConnectionError:
            raise RuntimeError("Server did not start or is unreachable")

    @classmethod
    def tearDownClass(cls):
        """Clean up server after all tests."""
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(cls.server_process.pid, signal.SIGKILL)

            for stream in [cls.server_process.stdin, cls.server_process.stdout, cls.server_process.stderr]:
                if stream and not stream.closed:
                    stream.close()

    def test_minesweeper_env_client(self):
        """Test Minesweeper environment client initialization."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        assert isinstance(client, MinesweeperEnv)

    def test_minesweeper_initial_state(self):
        """Test the initial state after reset."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        result = client.reset()
        observation = result.observation

        # Check observation type and attributes
        assert isinstance(observation, MinesweeperObservation)
        assert isinstance(observation.board, list)
        assert isinstance(observation.done, bool)
        assert isinstance(observation.reward, float)
        assert isinstance(observation.num_mines, int)
        assert isinstance(observation.flags_placed, int)
        assert isinstance(observation.cells_revealed, int)
        # game_status may be int or GameStatus enum due to HTTP serialization
        assert isinstance(observation.game_status, (GameStatus, int))

        # Check initial state values
        assert observation.done is False
        assert observation.reward == 0.0
        assert observation.flags_placed == 0
        assert observation.cells_revealed == 0
        # Compare with enum value (handles both int and enum)
        assert observation.game_status == GameStatus.ONGOING.value or observation.game_status == GameStatus.ONGOING

        # Check board structure (default 5x5)
        assert len(observation.board) == 5  # 5 rows
        assert all(len(row) == 5 for row in observation.board)  # 5 columns

        # Check all cells are initially unrevealed
        assert all(
            cell == -1 for row in observation.board for cell in row
        ), "All cells should be unrevealed (-1) at start"

    def test_reveal_action(self):
        """Test revealing a cell."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Try revealing a cell
        action = MinesweeperAction(row=0, col=0, action_type="reveal")
        result = client.step(action)
        observation = result.observation

        assert isinstance(observation, MinesweeperObservation)
        assert observation.cells_revealed > 0, "At least one cell should be revealed"

    def test_flag_action(self):
        """Test placing a flag."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Place a flag
        action = MinesweeperAction(row=1, col=1, action_type="flag")
        result = client.step(action)
        observation = result.observation

        assert isinstance(observation, MinesweeperObservation)
        assert observation.flags_placed == 1, "One flag should be placed"
        assert observation.board[1][1] == "F", "Cell should show flag marker"

    def test_toggle_flag(self):
        """Test toggling a flag on and off."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Place a flag
        action = MinesweeperAction(row=2, col=2, action_type="flag")
        result = client.step(action)
        observation = result.observation
        assert observation.flags_placed == 1

        # Remove the flag
        action = MinesweeperAction(row=2, col=2, action_type="flag")
        result = client.step(action)
        observation = result.observation
        assert observation.flags_placed == 0, "Flag should be removed"

    def test_invalid_position(self):
        """Test action with invalid position."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Try invalid row
        action = MinesweeperAction(row=10, col=0, action_type="reveal")
        result = client.step(action)
        observation = result.observation

        assert observation.reward < 0, "Should receive negative reward for invalid action"

        # Try invalid column
        action = MinesweeperAction(row=0, col=10, action_type="reveal")
        result = client.step(action)
        observation = result.observation

        assert observation.reward < 0, "Should receive negative reward for invalid action"

    def test_reveal_already_revealed(self):
        """Test revealing an already revealed cell."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Reveal a cell - try multiple cells to ensure one gets revealed
        # (some cells might cascade reveal if they have 0 adjacent mines)
        action = MinesweeperAction(row=2, col=2, action_type="reveal")
        result = client.step(action)
        first_reward = result.observation.reward

        # Make sure the cell was actually revealed (could be revealed by cascade)
        # If it wasn't revealed successfully, try another cell
        if result.observation.board[2][2] == -1:
            action = MinesweeperAction(row=1, col=1, action_type="reveal")
            result = client.step(action)
            test_row, test_col = 1, 1
        else:
            test_row, test_col = 2, 2

        # Try revealing the same cell again
        action = MinesweeperAction(row=test_row, col=test_col, action_type="reveal")
        result = client.step(action)
        second_reward = result.observation.reward

        assert second_reward < 0, f"Should receive penalty for revealing already revealed cell, got {second_reward}"

    def test_game_status_ongoing(self):
        """Test that game status remains ONGOING during normal play."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Make a few safe moves
        action = MinesweeperAction(row=0, col=0, action_type="reveal")
        result = client.step(action)

        # Game should still be ongoing if we didn't hit a mine or win
        # Handle both int and enum types for game_status
        if (result.observation.game_status == GameStatus.ONGOING.value or
            result.observation.game_status == GameStatus.ONGOING):
            assert result.observation.done is False

    def test_board_cell_values(self):
        """Test that board cells contain valid values."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Reveal a cell
        action = MinesweeperAction(row=2, col=2, action_type="reveal")
        result = client.step(action)
        observation = result.observation

        # Check that revealed cells have valid values (0-8 or '*')
        for row in observation.board:
            for cell in row:
                assert (
                    cell == -1  # Unrevealed
                    or cell == "F"  # Flagged
                    or cell == "*"  # Mine (if revealed)
                    or (isinstance(cell, int) and 0 <= cell <= 8)  # Number of adjacent mines
                ), f"Invalid cell value: {cell}"

    def test_metadata_in_observation(self):
        """Test that observations contain metadata."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        result = client.reset()
        observation = result.observation

        assert hasattr(observation, "metadata"), "Observation should have metadata"
        assert isinstance(observation.metadata, dict), "Metadata should be a dictionary"

    def test_multiple_steps(self):
        """Test taking multiple steps in the environment."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        client.reset()

        # Take several actions
        actions = [
            MinesweeperAction(row=0, col=0, action_type="reveal"),
            MinesweeperAction(row=0, col=1, action_type="flag"),
            MinesweeperAction(row=1, col=0, action_type="reveal"),
        ]

        for action in actions:
            result = client.step(action)
            assert isinstance(result.observation, MinesweeperObservation)

    def test_reset_clears_state(self):
        """Test that reset properly clears the game state."""
        client = MinesweeperEnv(base_url="http://127.0.0.1:8000")

        # First game
        client.reset()
        action = MinesweeperAction(row=0, col=0, action_type="flag")
        client.step(action)

        # Reset and check state is cleared
        result = client.reset()
        observation = result.observation

        assert observation.flags_placed == 0, "Flags should be cleared after reset"
        assert observation.cells_revealed == 0, "Revealed cells should be cleared after reset"
        # Compare with enum value (handles both int and enum)
        assert (observation.game_status == GameStatus.ONGOING.value or
                observation.game_status == GameStatus.ONGOING), "Game should be ongoing after reset"


if __name__ == "__main__":
    unittest.main()
