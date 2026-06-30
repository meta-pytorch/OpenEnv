# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ground-truth scoring for pathway submissions (orchestrator-only)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


def normalize_label(text: str) -> str:
    """Lowercase alphanumeric tokens for fuzzy pathway / keyword matching."""
    s = (text or "").strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return " ".join(s.split())


def is_unknown_ground_truth(true_pathway: str) -> bool:
    t = normalize_label(true_pathway)
    return not t or t.startswith("unknown")


def _token_set(text: str) -> set[str]:
    return {t for t in normalize_label(text).split() if len(t) > 2}


def labels_match(a: str, b: str) -> bool:
    na, nb = normalize_label(a), normalize_label(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / min(len(ta), len(tb))
    return overlap >= 0.6


def keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    joined = normalize_label(text)
    hits: List[str] = []
    for kw in keywords:
        k = normalize_label(kw)
        if k and k in joined:
            hits.append(kw)
    return hits


# Base score awarded for a correct keyword-rubric (GEO) identification. The
# remaining ``1 - KEYWORD_BASE_SCORE`` is distributed by how many expected
# keywords the hypothesis hits. This keeps a correct GEO answer on a scale
# comparable to a correct exact-label answer (1.0) instead of collapsing to a
# small fraction such as 1/5 = 0.2, which otherwise biases leaderboards and
# RL advantage estimates across heterogeneous cases.
KEYWORD_BASE_SCORE = 0.7


def score_submission(
    hypothesis: str,
    *,
    true_pathway: str,
    expected_keywords: Optional[Sequence[str]] = None,
    pathway_gene_set_names: Optional[Sequence[str]] = None,
    true_pathway_aliases: Optional[Sequence[str]] = None,
    top_ora_pathways: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Score a submitted pathway hypothesis without exposing labels to agents.

    Returns dict with ``correct``, ``score`` (0–1), ``match_mode``, and details.

    Scoring is intentionally strict about *which* label earns full credit:

    * With a known ``true_pathway``, only that label (and any explicit
      ``true_pathway_aliases``) scores 1.0. Distractor pathways present in the
      case (``pathway_gene_set_names``) and arbitrary top ORA hits do NOT earn
      credit — naming a distractor that happens to be defined in the case is a
      reward-hacking surface, not a correct answer.
    * With keyword rubrics (GEO / theme-based cases), any keyword hit is
      correct, scored on a normalized scale (see ``KEYWORD_BASE_SCORE``).
    * Only when ground truth is genuinely unknown is the top ORA hit accepted.

    ``pathway_gene_set_names`` is retained for telemetry/back-compat but no
    longer grants credit on its own.
    """
    hyp = (hypothesis or "").strip()
    if not hyp:
        return {
            "correct": False,
            "score": 0.0,
            "match_mode": "empty_hypothesis",
            "matched_label": None,
        }

    keywords = list(expected_keywords or [])
    ora_names = list(top_ora_pathways or [])

    # Keyword rubric (GEO / theme-based cases).
    if keywords:
        hits = keyword_hits(hyp, keywords)
        if hits:
            extra_fraction = len(hits) / max(1, len(keywords))
            score = KEYWORD_BASE_SCORE + (1.0 - KEYWORD_BASE_SCORE) * extra_fraction
            return {
                "correct": True,
                "score": round(min(1.0, score), 4),
                "match_mode": "expected_keywords",
                "matched_label": hits[0],
                "keyword_hits": hits,
            }

    # Known ground truth: credit only the true pathway (and explicit aliases).
    if not is_unknown_ground_truth(true_pathway):
        candidates = [true_pathway, *(true_pathway_aliases or [])]
        seen: set[str] = set()
        for label in candidates:
            key = normalize_label(label)
            if not key or key in seen:
                continue
            seen.add(key)
            if labels_match(hyp, label):
                return {
                    "correct": True,
                    "score": 1.0,
                    "match_mode": "pathway_label",
                    "matched_label": label,
                }
        # Known truth but no match: incorrect. Do not credit distractor
        # pathways or top ORA hits.
        return {
            "correct": False,
            "score": 0.0,
            "match_mode": "no_match",
            "matched_label": None,
        }

    # Unknown ground truth: accept top ORA hit if agent names it exactly.
    if ora_names and labels_match(hyp, ora_names[0]):
        return {
            "correct": True,
            "score": 0.85,
            "match_mode": "top_ora_pathway",
            "matched_label": ora_names[0],
        }

    return {
        "correct": False,
        "score": 0.0,
        "match_mode": "no_match",
        "matched_label": None,
    }
