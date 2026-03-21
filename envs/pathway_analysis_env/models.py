# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Pathway Analysis Environment.

This module defines the Action, Observation, and State types for a toy
pathway-inference task. An agent inspects omics metadata, runs differential
expression and pathway enrichment analyses, then submits a hypothesis about
which signaling pathway is activated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class PathwayAction(Action):
    """
    Action for the Pathway Analysis environment.

    Attributes:
        action_type: One of "inspect_dataset", "run_differential_expression",
            "run_pathway_enrichment", or "submit_answer".
        condition_a: Optional condition label for DE comparison.
        condition_b: Optional condition label for DE comparison.
        gene_list: Optional gene list for targeted enrichment.
        hypothesis: Pathway name to submit as the answer.
    """

    action_type: str
    condition_a: Optional[str] = None
    condition_b: Optional[str] = None
    gene_list: Optional[List[str]] = None
    hypothesis: Optional[str] = None


class PathwayObservation(Observation):
    """
    Observation for the Pathway Analysis environment.

    Inherits ``done``, ``reward``, and ``metadata`` from the base
    ``Observation`` class.

    Attributes:
        message: Human-readable description of the observation.
        available_conditions: Condition labels present in the dataset.
        top_genes: Top differentially expressed genes (populated after DE).
        top_pathways: Top enriched pathways (populated after enrichment).
    """

    message: str = ""
    available_conditions: List[str] = Field(default_factory=list)
    top_genes: List[str] = Field(default_factory=list)
    top_pathways: List[str] = Field(default_factory=list)


class PathwayState(State):
    """
    State for the Pathway Analysis environment.

    Attributes:
        episode_id: Unique ID for the current episode.
        step_count: Number of steps taken so far.
        true_pathway: The hidden ground-truth pathway for this episode.
        conditions: Condition labels in the loaded dataset.
        de_run: Whether differential expression has been executed.
        enrichment_run: Whether pathway enrichment has been executed.
        is_done: Whether the episode has ended.
    """

    true_pathway: str = ""
    conditions: List[str] = Field(default_factory=list)
    de_run: bool = False
    enrichment_run: bool = False
    is_done: bool = False
