# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenAI-style tool definitions and mapping to PathwayAction.

Usage:
    from pathway_analysis_env.agent_openai_tools import (
        OPENAI_TOOLS,
        tool_call_to_pathway_action,
    )

    # Pass OPENAI_TOOLS to OpenAI Chat Completions `tools=` or Responses API.
    # On tool_call, convert and step:
    action = tool_call_to_pathway_action(
        name=tool_call.function.name,
        arguments_json=tool_call.function.arguments,
    )
    result = await client.step(action)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from pathway_analysis_env.models import PathwayAction

_TOOL_NAMES: Set[str] = {
    "understand_experiment_design",
    "inspect_dataset",
    "run_differential_expression",
    "run_pathway_enrichment",
    "compare_pathways",
    "submit_answer",
    "pathway_env_step",
}


# Per-action tools (recommended). Load from JSON to keep a single source of truth.
def _load_tools() -> List[Dict[str, Any]]:
    path = Path(__file__).with_name("agent_openai_tools.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tools"])


OPENAI_TOOLS: List[Dict[str, Any]] = _load_tools()


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _coerce_gene_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("gene_list must be an array of strings")
    return [str(g).strip() for g in value if str(g).strip()]


def tool_call_to_pathway_action(
    *,
    name: str,
    arguments_json: str | Mapping[str, Any],
) -> PathwayAction:
    """
    Convert an OpenAI tool call into a PathwayAction for env.step().

    Supports:
      - Six named tools (name = action_type)
      - Unified ``pathway_env_step`` with action_type inside arguments
    """
    if isinstance(arguments_json, str):
        parsed: Any = json.loads(arguments_json) if arguments_json.strip() else {}
    else:
        parsed = arguments_json
    # Models occasionally emit ``null`` / non-object arguments (e.g. the JSON
    # literal ``null`` parses to ``None``). Treat anything that is not a
    # mapping as empty so callers fail gracefully instead of raising
    # ``AttributeError`` on ``args.get(...)``.
    args: Dict[str, Any] = dict(parsed) if isinstance(parsed, Mapping) else {}

    if name == "pathway_env_step":
        action_type = args.get("action_type")
        if not action_type or action_type not in _TOOL_NAMES - {"pathway_env_step"}:
            raise ValueError(
                f"Invalid action_type in pathway_env_step: {action_type!r}"
            )
    elif name in _TOOL_NAMES:
        action_type = name
    else:
        raise ValueError(f"Unknown tool name: {name!r}")

    return PathwayAction(
        action_type=action_type,
        condition_a=_coerce_optional_str(args.get("condition_a")),
        condition_b=_coerce_optional_str(args.get("condition_b")),
        gene_list=_coerce_gene_list(args.get("gene_list")),
        hypothesis=_coerce_optional_str(args.get("hypothesis")),
        pathway_a=_coerce_optional_str(args.get("pathway_a")),
        pathway_b=_coerce_optional_str(args.get("pathway_b")),
    )


# Default row caps for list-valued observation fields. Large omics tables
# (differential expression results, enrichment rows) otherwise balloon the LLM
# context and burn tokens; the agent only needs the top-ranked rows to reason,
# and the environment scores against its own full internal tables regardless.
_OBS_LIST_CAPS: Dict[str, int] = {
    "de_genes": 30,
    "pathway_enrichment": 20,
    "top_genes": 30,
    "top_pathways": 20,
}


def truncate_observation_payload(
    payload: Dict[str, Any],
    *,
    list_caps: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """
    Cap long list-valued fields in an observation payload to control token use.

    Returns a shallow copy with capped lists. A ``_truncation_note`` field is
    added when anything was truncated so the agent knows results were trimmed.
    The local ``trace_path`` is dropped (not useful to a remote agent).
    """
    caps = dict(_OBS_LIST_CAPS)
    if list_caps:
        caps.update(list_caps)
    out = dict(payload)
    notes: List[str] = []
    for key, cap in caps.items():
        val = out.get(key)
        if isinstance(val, list) and len(val) > cap:
            notes.append(f"{key}: showing top {cap} of {len(val)}")
            out[key] = val[:cap]
    out.pop("trace_path", None)
    if notes:
        out["_truncation_note"] = "; ".join(notes)
    return out


def observation_to_tool_result_content(
    observation: Any,
    *,
    truncate: bool = True,
    list_caps: Optional[Mapping[str, int]] = None,
) -> str:
    """Serialize observation for OpenAI tool role message (truncation-friendly).

    By default, long omics tables are capped (see ``truncate_observation_payload``)
    to keep tool results within practical context/token budgets. Pass
    ``truncate=False`` to serialize the full payload.
    """
    if hasattr(observation, "model_dump"):
        payload = observation.model_dump()
    elif isinstance(observation, dict):
        payload = observation
    else:
        payload = {
            "message": getattr(observation, "message", ""),
            "reward": getattr(observation, "reward", 0.0),
            "done": getattr(observation, "done", False),
            "metadata": getattr(observation, "metadata", {}),
        }
    if truncate:
        payload = truncate_observation_payload(payload, list_caps=list_caps)
    return json.dumps(payload, default=str)
