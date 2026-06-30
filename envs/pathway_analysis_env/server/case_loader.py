# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load pathway cases with optional separation of agent-visible vs orchestrator secrets."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Keys that must not ship to untrusted agent runtimes (orchestrator keeps full case).
CASE_SECRET_KEYS = frozenset(
    {
        "true_pathway",
        "true_pathway_aliases",
        "expected_keywords",
        "expert_hint",
        "expert_penalty",
        "expert_budget",
    }
)


def strip_case_secrets(case: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``case`` without orchestrator-only fields."""
    out = deepcopy(case)
    for key in CASE_SECRET_KEYS:
        out.pop(key, None)
    return out


def extract_case_secrets(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "true_pathway": str(case.get("true_pathway", "")),
        "true_pathway_aliases": list(case.get("true_pathway_aliases") or []),
        "expected_keywords": list(case.get("expected_keywords") or []),
        "expert_hint": case.get("expert_hint"),
        "expert_budget": case.get("expert_budget"),
        "expert_penalty": case.get("expert_penalty"),
    }


def load_case_file(
    data_dir: Path,
    case_name: str,
    *,
    agent_safe: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load case JSON from ``data_dir``.

    Returns ``(case_dict, secrets)``. When ``agent_safe`` is True, ``case_dict`` has
    secret keys removed (for agent containers); secrets are still returned for the server.
    """
    path = data_dir / case_name
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    secrets = extract_case_secrets(raw)
    if agent_safe:
        return strip_case_secrets(raw), secrets
    return raw, secrets


def export_agent_safe_case(
    source: Path,
    destination: Path,
) -> None:
    """Write an agent-safe case JSON (no ground-truth fields)."""
    with open(source, "r", encoding="utf-8") as f:
        raw = json.load(f)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(strip_case_secrets(raw), f, indent=2)
        f.write("\n")
