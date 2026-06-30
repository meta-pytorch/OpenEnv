# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

from pathway_analysis_env.server.case_loader import (
    CASE_SECRET_KEYS,
    load_case_file,
    strip_case_secrets,
)
from pathway_analysis_env.server.pathway_environment import DATA_DIR, PathwayEnvironment
from pathway_analysis_env.models import PathwayAction


def test_strip_case_secrets():
    raw = {
        "case_id": "x",
        "true_pathway": "MAPK signaling",
        "expected_keywords": ["mapk"],
        "counts": {"G1": [1, 2]},
    }
    pub = strip_case_secrets(raw)
    for k in CASE_SECRET_KEYS:
        assert k not in pub
    assert "counts" in pub


def test_reset_agent_safe_case_has_no_secrets_in_memory():
    env = PathwayEnvironment(case_file="toy_case_001.json")
    env.reset(eval_mode=True, orchestrator_mode=False)
    dumped = json.dumps(env._case)
    assert "true_pathway" not in dumped


def test_episode_observation_no_correct_without_orchestrator():
    env = PathwayEnvironment(case_file="toy_case_legacy.json")
    env.reset()
    env.step(PathwayAction(action_type="run_differential_expression"))
    env.step(PathwayAction(action_type="run_pathway_enrichment"))
    fin = env.step(
        PathwayAction(action_type="submit_answer", hypothesis="MAPK signaling")
    )
    assert fin.metadata.get("correct") is None
    assert env.episode_outcome.get("correct") is True


def test_load_case_file_roundtrip():
    case, secrets = load_case_file(DATA_DIR, "toy_case_001.json", agent_safe=False)
    assert secrets["true_pathway"] == "MAPK signaling"
    assert case["case_id"]
