# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Wordle Environment.

A word guessing game where players try to guess a 5-letter word within 6 attempts.
Players receive feedback on letter correctness and positioning.
"""

from .client import WordleEnv
from .models import (
    LetterFeedback,
    LetterStatus,
    WordleAction,
    WordleObservation,
    WordleState,
)

__all__ = [
    "LetterFeedback",
    "LetterStatus",
    "WordleAction",
    "WordleObservation",
    "WordleState",
    "WordleEnv",
]
