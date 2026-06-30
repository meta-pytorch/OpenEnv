# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pathway Analysis Environment Client.

Provides a WebSocket-based client for interacting with a running
Pathway Analysis server.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import PathwayAction, PathwayObservation, PathwayState


class PathwayEnv(EnvClient[PathwayAction, PathwayObservation, PathwayState]):
    """
    Client for the Pathway Analysis Environment.

    Example:
        >>> with PathwayEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(PathwayAction(action_type="inspect_dataset"))
        ...     print(result.observation.message)
    """

    def _step_payload(self, action: PathwayAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "condition_a": action.condition_a,
            "condition_b": action.condition_b,
            "gene_list": action.gene_list,
            "hypothesis": action.hypothesis,
            "pathway_a": action.pathway_a,
            "pathway_b": action.pathway_b,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PathwayObservation]:
        obs_data = payload.get("observation", {})
        observation = PathwayObservation(
            message=obs_data.get("message", ""),
            available_conditions=obs_data.get("available_conditions", []),
            top_genes=obs_data.get("top_genes", []),
            top_pathways=obs_data.get("top_pathways", []),
            de_genes=obs_data.get("de_genes", []),
            pathway_enrichment=obs_data.get("pathway_enrichment", []),
            pathway_comparison=obs_data.get("pathway_comparison"),
            overlap_summary=obs_data.get("overlap_summary"),
            statistical_ambiguity=obs_data.get("statistical_ambiguity"),
            trace_path=obs_data.get("trace_path"),
            experiment_design=obs_data.get("experiment_design"),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PathwayState:
        return PathwayState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            conditions=payload.get("conditions", []),
            de_run=payload.get("de_run", False),
            enrichment_run=payload.get("enrichment_run", False),
            is_done=payload.get("is_done", False),
            pipeline_mode=payload.get("pipeline_mode", False),
            strict_mode=payload.get("strict_mode", False),
            legacy_mode=payload.get("legacy_mode", False),
            eval_mode=payload.get("eval_mode", True),
            max_steps=payload.get("max_steps", 30),
            design_understood=payload.get("design_understood", False),
            validated_reference=payload.get("validated_reference"),
            validated_alternate=payload.get("validated_alternate"),
        )

    def _http_base_url(self) -> str:
        ws = getattr(self, "_ws_url", "ws://localhost:8000/ws")
        http = ws.replace("wss://", "https://").replace("ws://", "http://")
        if http.endswith("/ws"):
            http = http[:-3]
        return http.rstrip("/")

    def fetch_episode_outcome(self, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Fetch orchestrator episode score from a running pathway server.

        Requires the server started with web interface (default). Only valid after
        ``submit_answer`` in the same session.
        """
        base = self._http_base_url()
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/orchestrator/episode_outcome")
            r.raise_for_status()
            return r.json()

    def fetch_eval_protocol(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Fetch eval protocol summary from ``GET /orchestrator/eval_protocol``."""
        base = self._http_base_url()
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base}/orchestrator/eval_protocol")
            r.raise_for_status()
            return r.json()
