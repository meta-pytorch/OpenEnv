# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client for the Wordle environment.
"""

from core import HTTPEnvClient
from core import StepResult

from .models import LetterFeedback, LetterStatus, WordleAction, WordleObservation, WordleState


class WordleEnv(HTTPEnvClient[WordleAction, WordleObservation]):
    """
    HTTP client for the Wordle environment.
    
    This client handles communication with the Wordle environment server,
    including action submission and result parsing.
    """

    def _step_payload(self, action: WordleAction) -> dict:
        """
        Convert WordleAction to payload for HTTP request.
        
        Args:
            action: WordleAction to convert
            
        Returns:
            Dictionary payload for HTTP request
        """
        return {"guess": action.guess}

    def _parse_result(self, payload: dict) -> StepResult[WordleObservation]:
        """
        Parse HTTP response into StepResult.
        
        Args:
            payload: HTTP response payload
            
        Returns:
            StepResult containing observation, reward, and done status
        """
        obs_data = payload["observation"]
        
        # Parse feedback
        feedback = []
        for fb_data in obs_data.get("feedback", []):
            feedback.append(LetterFeedback(
                letter=fb_data["letter"],
                status=LetterStatus(fb_data["status"]),
                position=fb_data["position"]
            ))
        
        observation = WordleObservation(
            guess=obs_data["guess"],
            feedback=feedback,
            attempt_number=obs_data["attempt_number"],
            max_attempts=obs_data.get("max_attempts", 6),
            game_won=obs_data.get("game_won", False),
            game_lost=obs_data.get("game_lost", False),
            correct_word=obs_data.get("correct_word"),
            used_letters=obs_data.get("used_letters", []),
            correct_letters=obs_data.get("correct_letters", []),
            wrong_position_letters=obs_data.get("wrong_position_letters", []),
            not_in_word_letters=obs_data.get("not_in_word_letters", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            metadata=obs_data.get("metadata", {})
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> WordleState:
        """
        Parse state from HTTP response.
        
        Args:
            payload: HTTP response payload
            
        Returns:
            WordleState object
        """
        # Parse all feedback
        all_feedback = []
        for feedback_list in payload.get("all_feedback", []):
            feedback = []
            for fb_data in feedback_list:
                feedback.append(LetterFeedback(
                    letter=fb_data["letter"],
                    status=LetterStatus(fb_data["status"]),
                    position=fb_data["position"]
                ))
            all_feedback.append(feedback)
        
        return WordleState(
            episode_id=payload["episode_id"],
            step_count=payload["step_count"],
            target_word=payload.get("target_word", ""),
            attempt_number=payload.get("attempt_number", 0),
            max_attempts=payload.get("max_attempts", 6),
            game_won=payload.get("game_won", False),
            game_lost=payload.get("game_lost", False),
            guesses=payload.get("guesses", []),
            all_feedback=all_feedback,
            used_letters=payload.get("used_letters", []),
            correct_letters=payload.get("correct_letters", []),
            wrong_position_letters=payload.get("wrong_position_letters", []),
            not_in_word_letters=payload.get("not_in_word_letters", [])
        )
