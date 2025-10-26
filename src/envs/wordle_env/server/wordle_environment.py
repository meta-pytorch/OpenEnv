# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Wordle Environment Implementation.

A word guessing game where players try to guess a 5-letter word within 6 attempts.
Players receive feedback on letter correctness and positioning.
"""

import random
from uuid import uuid4
from typing import List

from core.env_server.interfaces import Environment

from ..models import (
    LetterFeedback,
    LetterStatus,
    WordleAction,
    WordleObservation,
    WordleState,
)


class WordleEnvironment(Environment):
    """
    Wordle game environment.
    
    Players try to guess a 5-letter word within 6 attempts.
    Each guess provides feedback on letter correctness and positioning.
    
    Example:
        >>> env = WordleEnvironment()
        >>> obs = env.reset()
        >>> print(obs.attempt_number)  # 1
        >>>
        >>> obs = env.step(WordleAction(guess="CRANE"))
        >>> print(obs.feedback[0].status)  # LetterStatus.CORRECT/WRONG_POSITION/NOT_IN_WORD
    """

    # Common 5-letter words for the game
    WORD_LIST = [
        "CRANE", "SLATE", "CRATE", "TRACE", "GRACE", "SPACE", "PLACE", "PEACE",
        "REACT", "TEACH", "REACH", "BEACH", "LEACH", "LEACH", "BREAK", "GREAT",
        "STEAM", "DREAM", "CREAM", "GLEAM", "STREAM", "SCREAM", "FRAME", "GAME",
        "FLAME", "BLAME", "SHAME", "FRAME", "PRAME", "CRAME", "DRAME", "TRAME",
        "WORLD", "WORDS", "WORKS", "WORRY", "WORSE", "WORST", "WORTH", "WORTH",
        "HOUSE", "MOUSE", "ROUSE", "LOUSE", "DOUSE", "SOUSE", "POUSE", "TOUSE",
        "LIGHT", "RIGHT", "FIGHT", "NIGHT", "MIGHT", "SIGHT", "TIGHT", "WIGHT",
        "BOUND", "FOUND", "ROUND", "SOUND", "POUND", "WOUND", "HOUND", "MOUND",
        "YOUNG", "YOUNG", "YOUNG", "YOUNG", "YOUNG", "YOUNG", "YOUNG", "YOUNG",
        "EARTH", "HEART", "LEARN", "YEARN", "EARLY", "PEARL", "WEARY", "TEARY",
        "STONE", "PHONE", "ALONE", "ATONE", "STONE", "THONE", "SHONE", "PRONE"
    ]

    def __init__(self, max_attempts: int = 6):
        """
        Initialize the Wordle environment.
        
        Args:
            max_attempts: Maximum number of attempts (default: 6)
        """
        self._max_attempts = max_attempts
        self._state = WordleState(
            episode_id=str(uuid4()),
            step_count=0,
            max_attempts=max_attempts
        )

    def reset(self) -> WordleObservation:
        """
        Reset the environment for a new game.
        
        Returns:
            WordleObservation with initial game state
        """
        # Select a random word
        target_word = random.choice(self.WORD_LIST).upper()
        
        self._state = WordleState(
            episode_id=str(uuid4()),
            step_count=0,
            target_word=target_word,
            attempt_number=0,
            max_attempts=self._max_attempts,
            game_won=False,
            game_lost=False,
            guesses=[],
            all_feedback=[],
            used_letters=[],
            correct_letters=[],
            wrong_position_letters=[],
            not_in_word_letters=[]
        )
        
        return WordleObservation(
            guess="",  # No guess yet
            feedback=[],  # No feedback yet
            attempt_number=0,
            max_attempts=self._max_attempts,
            game_won=False,
            game_lost=False,
            correct_word=None,
            used_letters=[],
            correct_letters=[],
            wrong_position_letters=[],
            not_in_word_letters=[],
            done=False,
            reward=0.0,
            metadata={"message": "New Wordle game started! Guess a 5-letter word."}
        )

    def step(self, action: WordleAction) -> WordleObservation:  # type: ignore[override]
        """
        Execute a step in the Wordle game.
        
        Args:
            action: WordleAction containing the player's guess
            
        Returns:
            WordleObservation with game results
        """
        self._state.step_count += 1
        self._state.attempt_number += 1
        
        guess = action.guess.upper().strip()
        
        # Validate guess
        if len(guess) != 5:
            return WordleObservation(
                guess=guess,
                feedback=[],
                attempt_number=self._state.attempt_number,
                max_attempts=self._max_attempts,
                game_won=False,
                game_lost=False,
                done=False,
                reward=-1.0,  # Penalty for invalid guess
                metadata={"error": "Guess must be exactly 5 letters long"}
            )
        
        if not guess.isalpha():
            return WordleObservation(
                guess=guess,
                feedback=[],
                attempt_number=self._state.attempt_number,
                max_attempts=self._max_attempts,
                game_won=False,
                game_lost=False,
                done=False,
                reward=-1.0,  # Penalty for invalid guess
                metadata={"error": "Guess must contain only letters"}
            )
        
        # Generate feedback
        feedback = self._generate_feedback(guess, self._state.target_word)
        
        # Update state
        self._state.guesses.append(guess)
        self._state.all_feedback.append(feedback)
        
        # Update letter tracking
        self._update_letter_tracking(guess, feedback)
        
        # Check win condition
        game_won = guess == self._state.target_word
        self._state.game_won = game_won
        
        # Check lose condition
        game_lost = not game_won and self._state.attempt_number >= self._max_attempts
        self._state.game_lost = game_lost
        
        # Calculate reward
        reward = self._calculate_reward(game_won, game_lost, feedback)
        
        # Determine if episode is done
        done = game_won or game_lost
        
        return WordleObservation(
            guess=guess,
            feedback=feedback,
            attempt_number=self._state.attempt_number,
            max_attempts=self._max_attempts,
            game_won=game_won,
            game_lost=game_lost,
            correct_word=self._state.target_word if done else None,
            used_letters=self._state.used_letters.copy(),
            correct_letters=self._state.correct_letters.copy(),
            wrong_position_letters=self._state.wrong_position_letters.copy(),
            not_in_word_letters=self._state.not_in_word_letters.copy(),
            done=done,
            reward=reward,
            metadata={
                "attempt": self._state.attempt_number,
                "max_attempts": self._max_attempts,
                "game_won": game_won,
                "game_lost": game_lost,
                "target_word": self._state.target_word if done else "hidden"
            }
        )

    def _generate_feedback(self, guess: str, target: str) -> List[LetterFeedback]:
        """Generate feedback for a guess."""
        feedback = []
        target_letters = list(target)
        guess_letters = list(guess)
        
        # First pass: mark correct positions
        for i, (g_letter, t_letter) in enumerate(zip(guess_letters, target_letters)):
            if g_letter == t_letter:
                feedback.append(LetterFeedback(
                    letter=g_letter,
                    status=LetterStatus.CORRECT,
                    position=i
                ))
                # Mark as used in target
                target_letters[i] = None
            else:
                feedback.append(LetterFeedback(
                    letter=g_letter,
                    status=LetterStatus.NOT_IN_WORD,  # Will be updated in second pass
                    position=i
                ))
        
        # Second pass: mark wrong positions
        for i, feedback_item in enumerate(feedback):
            if feedback_item.status == LetterStatus.NOT_IN_WORD:
                g_letter = feedback_item.letter
                if g_letter in target_letters:
                    feedback_item.status = LetterStatus.WRONG_POSITION
                    # Remove from target to avoid double counting
                    target_letters[target_letters.index(g_letter)] = None
        
        return feedback

    def _update_letter_tracking(self, guess: str, feedback: List[LetterFeedback]):
        """Update letter tracking based on feedback."""
        for letter in guess:
            if letter not in self._state.used_letters:
                self._state.used_letters.append(letter)
        
        for feedback_item in feedback:
            letter = feedback_item.letter
            if feedback_item.status == LetterStatus.CORRECT:
                if letter not in self._state.correct_letters:
                    self._state.correct_letters.append(letter)
            elif feedback_item.status == LetterStatus.WRONG_POSITION:
                if letter not in self._state.wrong_position_letters:
                    self._state.wrong_position_letters.append(letter)
            else:  # NOT_IN_WORD
                if letter not in self._state.not_in_word_letters:
                    self._state.not_in_word_letters.append(letter)

    def _calculate_reward(self, game_won: bool, game_lost: bool, feedback: List[LetterFeedback]) -> float:
        """Calculate reward based on game result and feedback."""
        if game_won:
            # Bonus for winning, more bonus for fewer attempts
            return 10.0 - (self._state.attempt_number - 1) * 0.5
        elif game_lost:
            return -5.0  # Penalty for losing
        else:
            # Reward based on correct letters
            correct_count = sum(1 for f in feedback if f.status == LetterStatus.CORRECT)
            wrong_position_count = sum(1 for f in feedback if f.status == LetterStatus.WRONG_POSITION)
            return correct_count * 1.0 + wrong_position_count * 0.5

    @property
    def state(self) -> WordleState:
        """
        Get the current environment state.
        
        Returns:
            Current WordleState with game information
        """
        return self._state
