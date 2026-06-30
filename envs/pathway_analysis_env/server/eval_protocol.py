# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Agent-safe evaluation protocol helpers for pathway_analysis_env."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from ..models import PathwayObservation

# Keys never sent to agents when eval_mode is on.
_AGENT_METADATA_BLOCKLIST = frozenset(
    {
        "correct",
        "static_top_genes",
        "static_top_pathways",
        "true_pathway",
        "ground_truth",
        "episode_score",
    }
)


def default_max_steps(case: Dict[str, Any]) -> int:
    return max(5, int(case.get("max_steps", 30)))


def resolve_eval_mode(case: Dict[str, Any], reset_kwargs: Dict[str, Any]) -> bool:
    """Eval mode is on unless reset(eval_mode=False) or case sets eval_mode: false."""
    if "eval_mode" in reset_kwargs:
        return bool(reset_kwargs["eval_mode"])
    return bool(case.get("eval_mode", True))


def resolve_orchestrator_mode(case: Dict[str, Any], reset_kwargs: Dict[str, Any]) -> bool:
    """Expose scoring details in metadata (for in-repo harnesses only)."""
    if "orchestrator_mode" in reset_kwargs:
        return bool(reset_kwargs["orchestrator_mode"])
    return bool(case.get("orchestrator_mode", False))


def shaping_reward(eval_mode: bool, nominal: float) -> float:
    """Zero intermediate shaping in eval mode; terminal scoring is separate."""
    if eval_mode:
        return 0.0
    return nominal


def sanitize_metadata_for_agent(
    metadata: Optional[Dict[str, Any]], *, eval_mode: bool
) -> Dict[str, Any]:
    if not metadata:
        return {}
    if not eval_mode:
        return dict(metadata)
    out = {k: v for k, v in metadata.items() if k not in _AGENT_METADATA_BLOCKLIST}
    return out


def sanitize_observation_for_agent(
    obs: PathwayObservation,
    *,
    eval_mode: bool,
    orchestrator_mode: bool,
    reward_override: Optional[float] = None,
) -> PathwayObservation:
    if not eval_mode:
        return obs
    meta = sanitize_metadata_for_agent(obs.metadata, eval_mode=True)
    if orchestrator_mode and obs.metadata and "correct" in obs.metadata:
        meta["correct"] = obs.metadata["correct"]
    if orchestrator_mode and obs.metadata and "episode_score" in obs.metadata:
        meta["episode_score"] = obs.metadata["episode_score"]
    reward = obs.reward if reward_override is None else reward_override
    if eval_mode and not orchestrator_mode:
        # Hide reward signal except strict terminal failures (negative).
        if obs.done and reward and reward > 0:
            reward = 0.0
        elif not obs.done:
            reward = 0.0
    return obs.model_copy(
        update={
            "metadata": meta,
            "reward": reward,
        }
    )


def strip_legacy_answer_leaks(
    inspect_meta: Dict[str, Any], *, eval_mode: bool
) -> Dict[str, Any]:
    if not eval_mode:
        return inspect_meta
    out = dict(inspect_meta)
    out.pop("static_top_genes", None)
    out.pop("static_top_pathways", None)
    return out
