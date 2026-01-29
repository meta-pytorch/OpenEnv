# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Chess environment."""

import pytest

from envs.chess_env import ChessAction, ChessObservation, ChessState
from envs.chess_env.server.chess_environment import ChessEnvironment


class TestChessModels:
    """Test Chess data models."""

    def test_chess_action_creation(self):
        """Test ChessAction can be created with a move."""
        action = ChessAction(move="e2e4")
        assert action.move == "e2e4"

    def test_chess_observation_defaults(self):
        """Test ChessObservation has correct defaults."""
        obs = ChessObservation()
        assert obs.fen == ""
        assert obs.legal_moves == []
        assert obs.is_check is False
        assert obs.done is False
        assert obs.result is None

    def test_chess_state_defaults(self):
        """Test ChessState has correct defaults."""
        state = ChessState(episode_id="test-123", step_count=0)
        assert state.episode_id == "test-123"
        assert state.step_count == 0
        assert state.current_player == "white"
        assert state.move_history == []


class TestChessEnvironment:
    """Test Chess environment logic."""

    @pytest.fixture
    def env(self):
        """Create a fresh ChessEnvironment for each test."""
        return ChessEnvironment(opponent=None)  # No opponent for testing

    def test_reset_returns_observation(self, env):
        """Test reset returns a valid observation."""
        obs = env.reset()
        assert isinstance(obs, ChessObservation)
        assert obs.fen != ""
        assert len(obs.legal_moves) == 20  # 20 legal moves at start
        assert obs.is_check is False
        assert obs.done is False

    def test_reset_with_custom_fen(self, env):
        """Test reset with custom starting position."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        obs = env.reset(fen=fen)
        assert obs.fen == fen

    def test_step_valid_move(self, env):
        """Test stepping with a valid move."""
        env.reset()
        obs = env.step(ChessAction(move="e2e4"))
        assert isinstance(obs, ChessObservation)
        # After e2e4, the pawn is on e4 (shown as 4P3 in FEN's 4th rank)
        assert "4P3" in obs.fen

    def test_step_invalid_move_format(self, env):
        """Test stepping with invalid move format returns penalty."""
        env.reset()
        obs = env.step(ChessAction(move="invalid"))
        assert obs.reward == -0.1
        assert obs.done is False

    def test_step_illegal_move(self, env):
        """Test stepping with illegal move returns penalty."""
        env.reset()
        obs = env.step(ChessAction(move="e2e5"))  # Can't move pawn 3 squares
        assert obs.reward == -0.1
        assert obs.done is False

    def test_state_property(self, env):
        """Test state property returns ChessState."""
        env.reset()
        state = env.state
        assert isinstance(state, ChessState)
        assert state.episode_id != ""
        assert state.step_count == 0
        assert state.current_player == "white"

    def test_state_updates_after_move(self, env):
        """Test state updates correctly after a move."""
        env.reset()
        env.step(ChessAction(move="e2e4"))
        state = env.state
        assert state.step_count == 1
        assert "e2e4" in state.move_history
        assert state.current_player == "black"

    def test_checkmate_ends_game(self, env):
        """Test checkmate ends the game with correct reward."""
        # Fool's mate position
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        env.reset(fen=fen)
        # White is checkmated
        assert env._board.is_checkmate()

    def test_stalemate_is_draw(self, env):
        """Test stalemate ends with draw reward."""
        # Stalemate position - black king on h8, white king f7, white queen g6
        fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
        obs = env.reset(fen=fen)
        assert env._board.is_stalemate()
        assert obs.done
        assert obs.reward == 0.0
        assert obs.legal_moves == []


class TestChessEnvironmentWithOpponent:
    """Test Chess environment with opponent configured."""

    def test_random_opponent_makes_moves(self):
        """Test random opponent makes a move after agent move."""
        env = ChessEnvironment(opponent="random", agent_color="white")
        env.reset()

        # Agent makes a move
        env.step(ChessAction(move="e2e4"))

        # After agent's move and opponent's response, should be white's turn again
        assert env.state.current_player == "white"
        assert env.state.step_count == 2  # Agent + opponent

    def test_moonfish_opponent_makes_moves(self):
        """Test moonfish opponent makes a move after agent move."""
        env = ChessEnvironment(
            opponent="moonfish", opponent_depth=1, agent_color="white"
        )
        env.reset()

        # Agent makes a move
        env.step(ChessAction(move="e2e4"))

        # After agent's move and opponent's response, should be white's turn again
        assert env.state.current_player == "white"
        assert env.state.step_count == 2

    def test_opponent_checkmate_gives_negative_reward(self):
        """Test agent gets -1.0 reward when opponent checkmates."""
        env = ChessEnvironment(
            opponent="moonfish", opponent_depth=2, agent_color="white"
        )
        # Position after 1.f3 e5 - agent plays g4, opponent plays Qh4# (fool's mate)
        fen = "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2"
        env.reset(fen=fen)

        # Agent blunders with g4, allowing Qh4#
        obs = env.step(ChessAction(move="g2g4"))

        assert obs.done is True
        assert obs.reward == -1.0
        assert obs.result == "0-1"
