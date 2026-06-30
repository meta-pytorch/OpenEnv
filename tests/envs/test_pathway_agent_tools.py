# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json

from pathway_analysis_env.agent_openai_tools import (
    observation_to_tool_result_content,
    tool_call_to_pathway_action,
    truncate_observation_payload,
)


def test_tool_call_with_null_arguments_does_not_crash():
    """Models sometimes emit the JSON literal ``null`` for tool arguments.

    ``json.loads("null")`` returns ``None``; the mapper must treat that as
    empty args instead of raising ``AttributeError`` on ``args.get(...)``.
    """
    action = tool_call_to_pathway_action(
        name="run_differential_expression", arguments_json="null"
    )
    assert action.action_type == "run_differential_expression"
    assert action.condition_a is None
    assert action.condition_b is None


def test_tool_call_with_empty_arguments():
    action = tool_call_to_pathway_action(name="inspect_dataset", arguments_json="")
    assert action.action_type == "inspect_dataset"


def test_tool_call_with_mapping_arguments():
    action = tool_call_to_pathway_action(
        name="submit_answer", arguments_json={"hypothesis": "MAPK signaling"}
    )
    assert action.action_type == "submit_answer"
    assert action.hypothesis == "MAPK signaling"


def test_tool_call_normal_json_arguments():
    action = tool_call_to_pathway_action(
        name="run_differential_expression",
        arguments_json='{"condition_a": "control", "condition_b": "treated"}',
    )
    assert action.condition_a == "control"
    assert action.condition_b == "treated"


def test_truncate_observation_payload_caps_long_lists():
    payload = {
        "message": "ok",
        "de_genes": [{"gene": f"G{i}"} for i in range(100)],
        "pathway_enrichment": [{"pathway": f"P{i}"} for i in range(50)],
        "trace_path": "/tmp/some/local/trace.html",
    }
    out = truncate_observation_payload(payload)
    assert len(out["de_genes"]) == 30
    assert len(out["pathway_enrichment"]) == 20
    assert "trace_path" not in out
    assert "_truncation_note" in out


def test_truncate_observation_payload_keeps_short_lists():
    payload = {"de_genes": [{"gene": "G1"}], "message": "ok"}
    out = truncate_observation_payload(payload)
    assert len(out["de_genes"]) == 1
    assert "_truncation_note" not in out


def test_observation_serialization_truncates_by_default():
    payload = {"de_genes": [{"gene": f"G{i}"} for i in range(100)], "message": "ok"}
    serialized = observation_to_tool_result_content(payload)
    restored = json.loads(serialized)
    assert len(restored["de_genes"]) == 30

    full = observation_to_tool_result_content(payload, truncate=False)
    assert len(json.loads(full)["de_genes"]) == 100
