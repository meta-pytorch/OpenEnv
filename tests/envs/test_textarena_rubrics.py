# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TextArena rubrics."""

import pytest
from typing import Any, Dict, List

from textarena_env.models import TextArenaAction, TextArenaMessage, TextArenaObservation
from textarena_env.rubrics import (
    WordleRubric,
    WordleGreensRubric,
    WordleYellowsRubric,
    WordleRepetitionsRubric,
    WordleCorrectRubric,
    build_rubric,
    extract_guess,
    extract_wordle_feedback,
    extract_feedback_counts,
)


def make_wordle_observation(
    feedback_text: str = "",
    reward: float = 0.0,
    done: bool = False,
) -> TextArenaObservation:
    """Create a mock Wordle observation."""
    messages = []
    if feedback_text:
        messages.append(
            TextArenaMessage(
                sender_id=-1,
                content=f"Feedback: {feedback_text}",
                category="MESSAGE",
            )
        )
    return TextArenaObservation(
        prompt="Guess the word",
        messages=messages,
        current_player_id=0,
        legal_players=[0],
        info={},
        reward=reward,
        done=done,
    )


class TestExtractGuess:
    """Test guess extraction from action text."""

    def test_bracketed_guess(self):
        assert extract_guess("[hello]") == "[hello]"
        assert extract_guess("[WORLD]") == "[world]"

    def test_unbracketed_guess(self):
        assert extract_guess("hello") == "[hello]"
        assert extract_guess("helloworld") == "[hello]"

    def test_short_text(self):
        assert extract_guess("hi") == "[dunno]"

    def test_mixed_text(self):
        assert extract_guess("My guess is [crane]") == "[crane]"


class TestExtractFeedbackCounts:
    """Test feedback parsing."""

    def test_standard_feedback(self):
        feedback = """crane
G Y X X G"""
        green, yellow = extract_feedback_counts(feedback)
        assert green == 2
        assert yellow == 1

    def test_all_green(self):
        feedback = """crane
G G G G G"""
        green, yellow = extract_feedback_counts(feedback)
        assert green == 5
        assert yellow == 0

    def test_no_feedback(self):
        green, yellow = extract_feedback_counts("")
        assert green == 0
        assert yellow == 0


class TestWordleGreensRubric:
    """Test green letter scoring."""

    def test_scores_greens(self):
        rubric = WordleGreensRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation(feedback_text="crane\nG G X X X")

        score = rubric(action, obs)
        assert score == pytest.approx(0.4)  # 2 greens = 2/5

    def test_all_greens(self):
        rubric = WordleGreensRubric()
        action = TextArenaAction(message="[hello]")
        obs = make_wordle_observation(feedback_text="hello\nG G G G G")

        score = rubric(action, obs)
        assert score == pytest.approx(1.0)

    def test_no_feedback(self):
        rubric = WordleGreensRubric()
        action = TextArenaAction(message="[hello]")
        obs = make_wordle_observation()

        score = rubric(action, obs)
        assert score == 0.0


class TestWordleRepetitionsRubric:
    """Test repetition penalty."""

    def test_first_guess_full_score(self):
        rubric = WordleRepetitionsRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation()

        score = rubric(action, obs)
        assert score == 1.0

    def test_repeated_guess_penalty(self):
        rubric = WordleRepetitionsRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation()

        rubric(action, obs)  # First time: 1.0
        score = rubric(action, obs)  # Second time: 0.0
        assert score == 0.0

        score = rubric(action, obs)  # Third time: -1.0
        assert score == -1.0

    def test_reset_clears_history(self):
        rubric = WordleRepetitionsRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation()

        rubric(action, obs)  # First time
        rubric(action, obs)  # Second time

        rubric.reset()

        score = rubric(action, obs)  # First time after reset
        assert score == 1.0


class TestWordleRubric:
    """Test composite Wordle rubric."""

    def test_evaluates_all_components(self):
        rubric = WordleRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation(
            feedback_text="crane\nG Y X X X",
            reward=0.0,
        )

        reward = rubric(action, obs)

        # Check all component scores are tracked
        assert rubric.greens.last_score == pytest.approx(0.2)  # 1 green
        assert rubric.yellows.last_score == pytest.approx(0.2)  # 1 yellow
        assert rubric.repetitions.last_score == 1.0  # first guess
        assert rubric.correct.last_score == 0.0  # not won yet

    def test_get_reward_signals(self):
        rubric = WordleRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation(
            feedback_text="crane\nG G G G G",
            reward=1.0,  # win
        )

        rubric(action, obs)
        signals = rubric.get_reward_signals()

        assert "wordle.greens" in signals
        assert "wordle.yellows" in signals
        assert "wordle.repetitions" in signals
        assert "wordle.correct" in signals

        assert signals["wordle.greens"] == pytest.approx(1.0)
        assert signals["wordle.correct"] == pytest.approx(1.0)

    def test_reset_clears_all(self):
        rubric = WordleRubric()
        action = TextArenaAction(message="[crane]")
        obs = make_wordle_observation()

        rubric(action, obs)
        rubric(action, obs)  # Repeat

        rubric.reset()

        # Repetitions should be fresh
        rubric(action, obs)
        assert rubric.repetitions.last_score == 1.0


class TestBuildRubric:
    """Test rubric factory."""

    def test_wordle_rubric(self):
        rubric = build_rubric("Wordle-v0")
        assert isinstance(rubric, WordleRubric)

    def test_unknown_env_returns_none(self):
        rubric = build_rubric("Unknown-v0")
        assert rubric is None
