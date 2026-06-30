# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from pathway_analysis_env.server.scoring import (
    is_unknown_ground_truth,
    KEYWORD_BASE_SCORE,
    labels_match,
    score_submission,
)


def test_labels_match_fuzzy():
    assert labels_match("MAPK signaling", "mapk signaling")
    assert labels_match(
        "Estrogen Response Early",
        "MSigDB_Hallmark_2020: Estrogen Response Early",
    )


def test_score_exact_pathway():
    out = score_submission(
        "MAPK signaling",
        true_pathway="MAPK signaling",
        pathway_gene_set_names=["MAPK signaling", "PI3K-Akt signaling"],
        top_ora_pathways=["ERK cascade"],
    )
    assert out["correct"] is True
    assert out["match_mode"] == "pathway_label"


def test_score_keywords_geo():
    out = score_submission(
        "Strong estrogen response hallmark",
        true_pathway="Unknown (GEO benchmark)",
        expected_keywords=["estrogen", "ESR1"],
        top_ora_pathways=[],
    )
    assert out["correct"] is True
    assert out["match_mode"] == "expected_keywords"


def test_unknown_ground_truth():
    assert is_unknown_ground_truth("Unknown (GEO benchmark)")


def test_distractor_pathway_label_not_credited():
    """Naming a distractor pathway defined in the case must not score 1.0.

    Regression guard for a reward-hacking surface: previously any pathway
    gene-set name present in the case (or any top ORA hit) matched as a
    ``pathway_label`` and earned full credit.
    """
    out = score_submission(
        "PI3K-Akt signaling",
        true_pathway="MAPK signaling",
        pathway_gene_set_names=["MAPK signaling", "ERK cascade", "PI3K-Akt signaling"],
        top_ora_pathways=["MAPK signaling", "ERK cascade", "PI3K-Akt signaling"],
    )
    assert out["correct"] is False
    assert out["score"] == 0.0
    assert out["match_mode"] == "no_match"


def test_top_ora_hit_not_credited_when_truth_known():
    """A top ORA hit that is not the true pathway must not earn credit."""
    out = score_submission(
        "ERK cascade",
        true_pathway="MAPK signaling",
        top_ora_pathways=["ERK cascade", "MAPK signaling"],
    )
    assert out["correct"] is False
    assert out["score"] == 0.0


def test_true_pathway_alias_credited():
    """Explicit aliases of the true pathway earn full credit."""
    out = score_submission(
        "MAPK/ERK pathway",
        true_pathway="MAPK signaling",
        true_pathway_aliases=["MAPK/ERK pathway", "RAS-MAPK"],
    )
    assert out["correct"] is True
    assert out["score"] == 1.0
    assert out["match_mode"] == "pathway_label"


def test_keyword_score_normalized_above_base():
    """A correct GEO answer scores on a scale comparable to exact matches.

    A single keyword hit out of several should land at or above the base
    score, not collapse to a small fraction like 1/5 = 0.2.
    """
    out = score_submission(
        "estrogen response",
        true_pathway="Unknown (GEO benchmark)",
        expected_keywords=["estrogen", "ESR1", "fulvestrant", "ER", "hormone"],
    )
    assert out["correct"] is True
    assert out["score"] >= KEYWORD_BASE_SCORE
    assert out["match_mode"] == "expected_keywords"
