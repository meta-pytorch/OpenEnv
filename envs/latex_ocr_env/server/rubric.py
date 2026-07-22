# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward computation for the LaTeX OCR environment.

The reward is a **weighted sum of named components**, each in ``[0, 1]``:

    reward = Σ w_i · component_i          (weights normalized to sum to 1)

Components:
  - ``edit_similarity``     — 1 − CER (normalized char edit distance), dense.
  - ``exact_match``         — 1.0 iff the canonical strings match exactly.
  - ``structural_validity`` — balanced delimiters (+ parseable LaTeX if
                              ``pylatexenc`` is available). Rewards well-formed output.
  - ``length_format``       — length agreement + format cleanliness (no code
                              fences). Discourages runaway / fenced output.

Weights default via env vars (``LATEX_OCR_W_EDIT`` / ``_EXACT`` / ``_STRUCT`` /
``_LENFMT``) and are renormalized so the reward always lands in ``[0, 1]``. All
sub-scores are returned for logging so training can introspect them.

Implemented with a pure-Python Levenshtein; ``pylatexenc`` is an optional
enhancement (structural check degrades gracefully to a balance check without it).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


_WHITESPACE = re.compile(r"\s+")

DEFAULT_WEIGHTS = {
    "edit_similarity": float(os.environ.get("LATEX_OCR_W_EDIT", "0.6")),
    "exact_match": float(os.environ.get("LATEX_OCR_W_EXACT", "0.2")),
    "structural_validity": float(os.environ.get("LATEX_OCR_W_STRUCT", "0.1")),
    "length_format": float(os.environ.get("LATEX_OCR_W_LENFMT", "0.1")),
}


def normalize_latex(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip."""
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


def _delimiters_balanced(text: str) -> bool:
    """Braces/brackets/parens nest correctly and \\left/\\right counts match."""
    opposite = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != opposite[ch]:
                return False
    if stack:
        return False
    return text.count(r"\left") == text.count(r"\right")


def _parseable(text: str) -> bool | None:
    """True/False if pylatexenc can walk the LaTeX; None if pylatexenc absent."""
    try:
        from pylatexenc.latexwalker import LatexWalker
    except Exception:
        return None
    try:
        LatexWalker(text).get_latex_nodes()
        return True
    except Exception:
        return False


@dataclass
class GradeResult:
    reward: float
    exact_match: bool
    char_error_rate: float
    components: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


class LatexOCRRubric:
    """Weighted-sum rubric over named reward components.

    Args:
        weights: component -> weight. Defaults from env vars; renormalized so the
            reward is always in ``[0, 1]``. Unknown keys are ignored; missing
            keys default to 0.
    """

    COMPONENTS = (
        "edit_similarity",
        "exact_match",
        "structural_validity",
        "length_format",
    )

    def __init__(self, weights: dict | None = None) -> None:
        raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
        for name, w in raw.items():
            if name in self.COMPONENTS and w < 0:
                raise ValueError(f"weight for {name} must be non-negative")
        picked = {k: float(raw.get(k, 0.0)) for k in self.COMPONENTS}
        total = sum(picked.values())
        if total <= 0:
            raise ValueError("at least one component weight must be positive")
        self.weights = {k: v / total for k, v in picked.items()}

    # --- individual components (each returns a score in [0, 1]) ---
    def _edit_and_exact(
        self, pred_canon: str, target_canon: str
    ) -> tuple[float, bool, float]:
        exact = pred_canon == target_canon
        if not target_canon:
            cer = 0.0 if not pred_canon else 1.0
        else:
            distance = levenshtein(pred_canon, target_canon)
            cer = min(1.0, distance / max(len(pred_canon), len(target_canon)))
        return 1.0 - cer, exact, cer

    def _structural(self, pred_norm: str) -> float:
        balanced = _delimiters_balanced(pred_norm)
        parseable = _parseable(pred_norm)
        if parseable is None:  # pylatexenc not installed -> balance only
            return 1.0 if balanced else 0.0
        return 0.5 * float(balanced) + 0.5 * float(parseable)

    def _length_format(
        self, prediction: str, pred_canon: str, target_canon: str
    ) -> float:
        fmt_clean = 0.0 if "```" in prediction else 1.0
        lp, lt = len(pred_canon), len(target_canon)
        length_ratio = 1.0 if max(lp, lt) == 0 else min(lp, lt) / max(lp, lt)
        return 0.5 * fmt_clean + 0.5 * length_ratio

    def grade(self, prediction: str, target: str) -> GradeResult:
        # Canonical (whitespace-stripped) forms: LaTeX spacing is cosmetic.
        pred_norm = normalize_latex(prediction)
        target_norm = normalize_latex(target)
        pred_canon = _strip_all_whitespace(pred_norm)
        target_canon = _strip_all_whitespace(target_norm)

        # An empty prediction against a non-empty target earns nothing: the
        # auxiliary components must not hand out floor credit for "no answer".
        if not pred_canon and target_canon:
            zero = {k: 0.0 for k in self.COMPONENTS}
            return GradeResult(
                reward=0.0,
                exact_match=False,
                char_error_rate=1.0,
                components=zero,
                weights={k: round(v, 6) for k, v in self.weights.items()},
            )

        edit_sim, exact, cer = self._edit_and_exact(pred_canon, target_canon)
        components = {
            "edit_similarity": edit_sim,
            "exact_match": float(exact),
            "structural_validity": self._structural(pred_norm),
            "length_format": self._length_format(prediction, pred_canon, target_canon),
        }
        reward = sum(self.weights[k] * components[k] for k in self.COMPONENTS)
        return GradeResult(
            reward=round(reward, 6),
            exact_match=exact,
            char_error_rate=round(cer, 6),
            components={k: round(v, 6) for k, v in components.items()},
            weights={k: round(v, 6) for k, v in self.weights.items()},
        )
