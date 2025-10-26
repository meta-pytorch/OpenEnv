# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Wordle Environment.

The Wordle environment is a word guessing game where players try to guess a 5-letter word
within 6 attempts, receiving feedback on letter correctness and positioning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from core.env_server.types import Action, Observation, State


class LetterStatus(Enum):
    """Status of a letter in the guess."""
    CORRECT = "correct"      # Letter is in the correct position
    WRONG_POSITION = "wrong_position"  # Letter is in the word but wrong position
    NOT_IN_WORD = "not_in_word"        # Letter is not in the word


@dataclass(kw_only=True)
class WordleAction(Action):
    """Action for the Wordle environment - player's word guess."""
    
    guess: str  # 5-letter word guess


@dataclass(kw_only=True)
class LetterFeedback:
    """Feedback for a single letter in the guess."""
    
    letter: str
    status: LetterStatus
    position: int


@dataclass(kw_only=True)
class WordleObservation(Observation):
    """Observation from the Wordle environment."""
    
    guess: str
    feedback: List[LetterFeedback]
    attempt_number: int
    max_attempts: int = 6
    game_won: bool = False
    game_lost: bool = False
    correct_word: Optional[str] = None  # Only revealed when game ends
    used_letters: List[str] = None  # Letters that have been used
    correct_letters: List[str] = None  # Letters in correct positions
    wrong_position_letters: List[str] = None  # Letters in wrong positions
    not_in_word_letters: List[str] = None  # Letters not in the word


@dataclass(kw_only=True)
class WordleState(State):
    """Extended state for Wordle environment."""
    
    target_word: str = ""
    attempt_number: int = 0
    max_attempts: int = 6
    game_won: bool = False
    game_lost: bool = False
    guesses: List[str] = None
    all_feedback: List[List[LetterFeedback]] = None
    used_letters: List[str] = None
    correct_letters: List[str] = None
    wrong_position_letters: List[str] = None
    not_in_word_letters: List[str] = None
    
    def __post_init__(self):
        if self.guesses is None:
            self.guesses = []
        if self.all_feedback is None:
            self.all_feedback = []
        if self.used_letters is None:
            self.used_letters = []
        if self.correct_letters is None:
            self.correct_letters = []
        if self.wrong_position_letters is None:
            self.wrong_position_letters = []
        if self.not_in_word_letters is None:
            self.not_in_word_letters = []
