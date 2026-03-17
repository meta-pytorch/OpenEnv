# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rubrics for TextArena environments.

This module provides Rubric implementations for TextArena games,
migrated from the legacy RewardProvider interface to the new
Rubric abstraction (RFC 004).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from openenv.core.rubrics import Rubric, RubricDict

try:
    from textarena_env.models import TextArenaAction, TextArenaObservation
except ImportError:
    from models import TextArenaAction, TextArenaObservation


_WORDLE_GUESS_PATTERN = re.compile(r"\[[A-Za-z]{5}\]")


def extract_guess(text: str) -> str:
    """Normalize a Wordle guess string from arbitrary text."""
    match = _WORDLE_GUESS_PATTERN.search(text)
    if match:
        return match.group(0).lower()

    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if len(cleaned) >= 5:
        return f"[{cleaned[:5]}]"
    return "[dunno]"


def extract_wordle_feedback(observation: TextArenaObservation) -> str:
    """Pull the latest feedback text from a Wordle observation."""
    for message in reversed(observation.messages):
        content = message.content.strip()
        if "Feedback:" in content:
            return content.split("Feedback:", 1)[-1].strip()
    return ""


def extract_feedback_counts(feedback: str) -> Tuple[int, int]:
    """Return counts of green (G) and yellow (Y) markers from feedback."""
    if not feedback:
        return (0, 0)

    lines = [line.strip() for line in feedback.split("\n") if line.strip()]
    if len(lines) < 2:
        return (0, 0)

    for line in reversed(lines):
        normalized = line.replace(" ", "")
        if normalized and all(c in "GYX" for c in normalized):
            green = normalized.count("G")
            yellow = normalized.count("Y")
            return (green, yellow)

    return (0, 0)


class WordleGreensRubric(Rubric):
    """Rubric that scores based on green (correct position) letters in Wordle.

    Returns a score from 0.0 to 1.0 based on how many letters are in
    the correct position (green count / 5).
    """

    def forward(self, action: Any, observation: Any) -> float:
        if not isinstance(observation, TextArenaObservation):
            return 0.0
        feedback = extract_wordle_feedback(observation)
        if not feedback:
            return 0.0
        green_count, _ = extract_feedback_counts(feedback)
        return green_count / 5.0


class WordleYellowsRubric(Rubric):
    """Rubric that scores based on yellow (correct letter, wrong position) in Wordle.

    Returns a score from 0.0 to 1.0 based on how many letters are in
    the word but wrong position (yellow count / 5).
    """

    def forward(self, action: Any, observation: Any) -> float:
        if not isinstance(observation, TextArenaObservation):
            return 0.0
        feedback = extract_wordle_feedback(observation)
        if not feedback:
            return 0.0
        _, yellow_count = extract_feedback_counts(feedback)
        return yellow_count / 5.0


class WordleRepetitionsRubric(Rubric):
    """Rubric that penalizes repeated guesses in Wordle.

    Returns 1.0 for first occurrence of a guess, decreasing by 1.0
    for each repetition (can go negative for heavily repeated guesses).
    """

    def __init__(self):
        super().__init__()
        self._guess_history: Dict[str, int] = {}

    def forward(self, action: Any, observation: Any) -> float:
        if not isinstance(action, TextArenaAction):
            return 0.0

        guess = extract_guess(action.message)
        normalized_guess = guess if guess and guess != "[dunno]" else ""

        if not normalized_guess:
            return 0.0

        previous_occurrences = self._guess_history.get(normalized_guess, 0)
        self._guess_history[normalized_guess] = previous_occurrences + 1

        return 1.0 - previous_occurrences

    def reset(self) -> None:
        """Clear guess history for new episode."""
        self._guess_history.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {"guess_history": dict(self._guess_history)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "guess_history" in state:
            self._guess_history = dict(state["guess_history"])


class WordleCorrectRubric(Rubric):
    """Rubric that returns the game's correct/win reward from observation.

    Simply passes through the reward from the underlying TextArena game.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if not isinstance(observation, TextArenaObservation):
            return 0.0
        return float(observation.reward or 0.0)


class WordleRubric(Rubric):
    """Composite rubric for Wordle that combines multiple scoring signals.

    Provides the same signals as the legacy _WordleRewardProvider:
    - greens: Score based on correct position letters
    - yellows: Score based on correct letters in wrong position
    - repetitions: Penalty for repeated guesses
    - correct: Pass-through of game win/loss reward

    Usage:
        rubric = WordleRubric()
        reward = rubric(action, observation)

        # Access individual component scores
        print(f"Greens: {rubric.greens.last_score}")
        print(f"Yellows: {rubric.yellows.last_score}")
    """

    SIGNAL_MAP = {
        "greens": "wordle.greens",
        "yellows": "wordle.yellows",
        "repetitions": "wordle.repetitions",
        "correct": "wordle.correct",
    }

    def __init__(self):
        super().__init__()
        self.greens = WordleGreensRubric()
        self.yellows = WordleYellowsRubric()
        self.repetitions = WordleRepetitionsRubric()
        self.correct = WordleCorrectRubric()

    def forward(self, action: Any, observation: Any) -> float:
        """Evaluate all sub-rubrics and return the correct (win) score."""
        # Evaluate all components - their scores are tracked via last_score
        self.greens(action, observation)
        self.yellows(action, observation)
        self.repetitions(action, observation)
        correct_score = self.correct(action, observation)

        # Return the game's win/loss reward as the primary signal
        return correct_score

    def reset(self) -> None:
        """Reset all child rubrics."""
        self.greens.reset()
        self.yellows.reset()
        self.repetitions.reset()
        self.correct.reset()

    def get_reward_signals(self) -> Dict[str, float]:
        """Get all reward signals in the legacy format.

        Returns a dict mapping signal names to scores, compatible with
        the legacy RewardProvider interface.
        """
        return {
            self.SIGNAL_MAP["greens"]: self.greens.last_score or 0.0,
            self.SIGNAL_MAP["yellows"]: self.yellows.last_score or 0.0,
            self.SIGNAL_MAP["repetitions"]: self.repetitions.last_score or 0.0,
            self.SIGNAL_MAP["correct"]: self.correct.last_score or 0.0,
        }


def build_rubric(env_id: str) -> Rubric | None:
    """Instantiate the appropriate rubric for the given environment.

    Args:
        env_id: The TextArena environment ID (e.g., "Wordle-v0").

    Returns:
        A Rubric instance for the environment, or None if no specific
        rubric is defined.
    """
    if env_id == "Wordle-v0":
        return WordleRubric()
    return None


__all__ = [
    "WordleRubric",
    "WordleGreensRubric",
    "WordleYellowsRubric",
    "WordleRepetitionsRubric",
    "WordleCorrectRubric",
    "build_rubric",
    "extract_guess",
    "extract_wordle_feedback",
    "extract_feedback_counts",
]
