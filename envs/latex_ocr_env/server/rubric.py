# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward computation for the LaTeX OCR environment.

The reward blends a dense similarity signal with an exact-match bonus, which is
the standard shape for OCR-style RL where exact-string reward is too sparse:

    reward = (1 - exact_weight) * (1 - CER) + exact_weight * exact_match

where ``CER`` is the normalized character error rate (Levenshtein distance over
the longer normalized string). A perfect transcription scores 1.0; a partial
transcription is capped at ``1 - exact_weight`` so the policy is always pulled
toward getting the expression *exactly* right.

Implemented with a pure-Python Levenshtein so the environment carries no extra
runtime dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WHITESPACE = re.compile(r"\s+")


def normalize_latex(text: str) -> str:
    """Canonicalize LaTeX for comparison.

    Collapses runs of whitespace to single spaces and strips leading/trailing
    space. Deliberately conservative: it does not reorder tokens or rewrite
    commands, so it never inflates the score by "fixing" a wrong answer.
    """
    return _WHITESPACE.sub(" ", text or "").strip()


def _strip_all_whitespace(text: str) -> str:
    return _WHITESPACE.sub("", text or "")


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (pure Python, O(len(a) * len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


@dataclass
class GradeResult:
    reward: float
    exact_match: bool
    char_error_rate: float


class LatexOCRRubric:
    """Normalized edit distance + exact-match bonus.

    Args:
        exact_weight: Fraction of the reward reserved for an exact match. With
            the default 0.2, partial answers score in ``[0, 0.8]`` and only an
            exact (whitespace-insensitive) match reaches ``1.0``.
    """

    def __init__(self, exact_weight: float = 0.2) -> None:
        if not 0.0 <= exact_weight <= 1.0:
            raise ValueError("exact_weight must be in [0, 1]")
        self.exact_weight = exact_weight

    def grade(self, prediction: str, target: str) -> GradeResult:
        # LaTeX spacing is largely cosmetic, so both the dense (edit-distance)
        # term and the exact-match term score on the whitespace-stripped form.
        # This keeps them consistent: an exact match => CER 0 => reward 1.0.
        pred_canon = _strip_all_whitespace(normalize_latex(prediction))
        target_canon = _strip_all_whitespace(normalize_latex(target))

        exact = pred_canon == target_canon

        if not target_canon:
            # Degenerate target: only an empty prediction is correct.
            cer = 0.0 if not pred_canon else 1.0
        else:
            distance = levenshtein(pred_canon, target_canon)
            cer = min(1.0, distance / max(len(pred_canon), len(target_canon)))

        similarity = 1.0 - cer
        reward = (1.0 - self.exact_weight) * similarity + self.exact_weight * float(
            exact
        )
        return GradeResult(
            reward=round(reward, 6),
            exact_match=exact,
            char_error_rate=round(cer, 6),
        )
