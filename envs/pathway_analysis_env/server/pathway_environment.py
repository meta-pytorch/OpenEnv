# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pathway Analysis Environment Implementation.

A toy computational-biology environment where an agent must identify which
signaling pathway is activated in a synthetic omics dataset.  Episode data
is deterministic and loaded from small JSON fixtures so the first version
stays lightweight and easy to debug.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from openenv.core.env_server import Environment

from ..models import PathwayAction, PathwayObservation, PathwayState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_case(case_name: str = "toy_case_001.json") -> Dict[str, Any]:
    with open(DATA_DIR / case_name, "r", encoding="utf-8") as f:
        return json.load(f)


class PathwayEnvironment(Environment):
    """
    Toy pathway-inference environment following the OpenEnv interface.

    Supports four actions:
    - inspect_dataset: view dataset metadata and available conditions.
    - run_differential_expression: obtain top differentially expressed genes.
    - run_pathway_enrichment: obtain top enriched pathways.
    - submit_answer: submit a pathway hypothesis and end the episode.
    """

    def __init__(self, case_file: str = "toy_case_001.json"):
        super().__init__()
        self._case_file = case_file
        self._case: Dict[str, Any] = {}
        self._state = PathwayState()
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PathwayObservation:
        self._case = load_case(self._case_file)
        self._state = PathwayState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            true_pathway=self._case["true_pathway"],
            conditions=self._case["conditions"],
        )
        return PathwayObservation(
            message=(
                "Toy omics dataset loaded. "
                "Inspect the dataset, run DE, run enrichment, or submit an answer."
            ),
            available_conditions=self._case["conditions"],
            metadata={"case_id": self._case["case_id"]},
        )

    def step(
        self,
        action: PathwayAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PathwayObservation:
        s = self._state
        s.step_count += 1

        if action.action_type == "inspect_dataset":
            return PathwayObservation(
                message="Dataset contains control and treated samples.",
                available_conditions=s.conditions,
                reward=0.1,
                metadata={"step_count": s.step_count},
            )

        if action.action_type == "run_differential_expression":
            s.de_run = True
            return PathwayObservation(
                message="Differential expression complete.",
                top_genes=self._case["top_genes"],
                reward=0.3,
                metadata={"step_count": s.step_count},
            )

        if action.action_type == "run_pathway_enrichment":
            s.enrichment_run = True
            return PathwayObservation(
                message="Pathway enrichment complete.",
                top_pathways=self._case["top_pathways"],
                reward=0.5,
                metadata={"step_count": s.step_count},
            )

        if action.action_type == "submit_answer":
            proposed = (action.hypothesis or "").strip().lower()
            correct = proposed == s.true_pathway.lower()
            s.is_done = True
            return PathwayObservation(
                message="Answer submitted.",
                done=True,
                reward=2.0 if correct else -1.0,
                metadata={"correct": correct, "step_count": s.step_count},
            )

        return PathwayObservation(
            message=f"Unknown action_type: {action.action_type}",
            reward=-0.2,
            metadata={"step_count": s.step_count},
        )

    @property
    def state(self) -> PathwayState:
        return self._state
