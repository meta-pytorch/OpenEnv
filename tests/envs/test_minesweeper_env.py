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

    def __init__(self, methodName="runTest"):
        self.client = None
        self.server_process = None
        super().__init__(methodName)

    def test_setup_server(self):
        """Set up the Minesweeper server for testing."""
        self.server_process = subprocess.Popen(
            ["python", "-m", "envs.minesweeper_env.server.app"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Give it a few seconds to start
        time.sleep(3)

    def check_server_running(self):
        """Check if the server is running and healthy."""
        try:
            response = requests.get("http://127.0.0.1:8000/health")
            self.assertEqual(response.status_code, 200)
        except requests.ConnectionError:
            self.fail("Server did not start or is unreachable")

    def test_minesweeper_env_client(self):
        """Test Minesweeper environment client initialization."""
        self.test_setup_server()
        self.check_server_running()

        self.client = MinesweeperEnv(base_url="http://127.0.0.1:8000")
        assert isinstance(self.client, MinesweeperEnv)

    def test_minesweeper_initial_state(self):
        """Test the initial state after reset."""
        self.test_minesweeper_env_client()

        result = self.client.reset()
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
        assert observation.done == False
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
        self.test_minesweeper_env_client()
        self.client.reset()

        # Try revealing a cell
        action = MinesweeperAction(row=0, col=0, action_type="reveal")
        result = self.client.step(action)
        observation = result.observation

        assert isinstance(observation, MinesweeperObservation)
        assert observation.cells_revealed > 0, "At least one cell should be revealed"

    def test_flag_action(self):
        """Test placing a flag."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Place a flag
        action = MinesweeperAction(row=1, col=1, action_type="flag")
        result = self.client.step(action)
        observation = result.observation

        assert isinstance(observation, MinesweeperObservation)
        assert observation.flags_placed == 1, "One flag should be placed"
        assert observation.board[1][1] == "F", "Cell should show flag marker"

    def test_toggle_flag(self):
        """Test toggling a flag on and off."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Place a flag
        action = MinesweeperAction(row=2, col=2, action_type="flag")
        result = self.client.step(action)
        observation = result.observation
        assert observation.flags_placed == 1

        # Remove the flag
        action = MinesweeperAction(row=2, col=2, action_type="flag")
        result = self.client.step(action)
        observation = result.observation
        assert observation.flags_placed == 0, "Flag should be removed"

    def test_invalid_position(self):
        """Test action with invalid position."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Try invalid row
        action = MinesweeperAction(row=10, col=0, action_type="reveal")
        result = self.client.step(action)
        observation = result.observation

        assert observation.reward < 0, "Should receive negative reward for invalid action"

        # Try invalid column
        action = MinesweeperAction(row=0, col=10, action_type="reveal")
        result = self.client.step(action)
        observation = result.observation

        assert observation.reward < 0, "Should receive negative reward for invalid action"

    def test_reveal_already_revealed(self):
        """Test revealing an already revealed cell."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Reveal a cell - try multiple cells to ensure one gets revealed
        # (some cells might cascade reveal if they have 0 adjacent mines)
        action = MinesweeperAction(row=2, col=2, action_type="reveal")
        result = self.client.step(action)
        first_reward = result.observation.reward

        # Make sure the cell was actually revealed (could be revealed by cascade)
        # If it wasn't revealed successfully, try another cell
        if result.observation.board[2][2] == -1:
            action = MinesweeperAction(row=1, col=1, action_type="reveal")
            result = self.client.step(action)
            test_row, test_col = 1, 1
        else:
            test_row, test_col = 2, 2

        # Try revealing the same cell again
        action = MinesweeperAction(row=test_row, col=test_col, action_type="reveal")
        result = self.client.step(action)
        second_reward = result.observation.reward

        assert second_reward < 0, f"Should receive penalty for revealing already revealed cell, got {second_reward}"

    def test_game_status_ongoing(self):
        """Test that game status remains ONGOING during normal play."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Make a few safe moves
        action = MinesweeperAction(row=0, col=0, action_type="reveal")
        result = self.client.step(action)

        # Game should still be ongoing if we didn't hit a mine or win
        # Handle both int and enum types for game_status
        if (result.observation.game_status == GameStatus.ONGOING.value or
            result.observation.game_status == GameStatus.ONGOING):
            assert result.observation.done == False

    def test_board_cell_values(self):
        """Test that board cells contain valid values."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Reveal a cell
        action = MinesweeperAction(row=2, col=2, action_type="reveal")
        result = self.client.step(action)
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
        self.test_minesweeper_env_client()
        result = self.client.reset()
        observation = result.observation

        assert hasattr(observation, "metadata"), "Observation should have metadata"
        assert isinstance(observation.metadata, dict), "Metadata should be a dictionary"

    def test_multiple_steps(self):
        """Test taking multiple steps in the environment."""
        self.test_minesweeper_env_client()
        self.client.reset()

        # Take several actions
        actions = [
            MinesweeperAction(row=0, col=0, action_type="reveal"),
            MinesweeperAction(row=0, col=1, action_type="flag"),
            MinesweeperAction(row=1, col=0, action_type="reveal"),
        ]

        for action in actions:
            result = self.client.step(action)
            assert isinstance(result.observation, MinesweeperObservation)

    def test_reset_clears_state(self):
        """Test that reset properly clears the game state."""
        self.test_minesweeper_env_client()

        # First game
        self.client.reset()
        action = MinesweeperAction(row=0, col=0, action_type="flag")
        self.client.step(action)

        # Reset and check state is cleared
        result = self.client.reset()
        observation = result.observation

        assert observation.flags_placed == 0, "Flags should be cleared after reset"
        assert observation.cells_revealed == 0, "Revealed cells should be cleared after reset"
        # Compare with enum value (handles both int and enum)
        assert (observation.game_status == GameStatus.ONGOING.value or
                observation.game_status == GameStatus.ONGOING), "Game should be ongoing after reset"

    def tearDown(self):
        """Clean up after tests."""
        if self.server_process:
            # Try terminating the process gracefully
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(self.server_process.pid, signal.SIGKILL)

            # Close the pipes to avoid ResourceWarnings
            for stream in [
                self.server_process.stdin,
                self.server_process.stdout,
                self.server_process.stderr,
            ]:
                if stream and not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
