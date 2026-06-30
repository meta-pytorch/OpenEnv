# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pathway analysis environment: PyDESeq2 DE, Fisher ORA, overlap-aware tools,
HTML episode trace.
"""

from __future__ import annotations

import html
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from ..models import PathwayAction, PathwayObservation, PathwayState
from . import failure_codes as FC
from .case_loader import load_case_file, strip_case_secrets
from .eval_protocol import (
    default_max_steps,
    resolve_eval_mode,
    resolve_orchestrator_mode,
    sanitize_observation_for_agent,
    shaping_reward,
    strip_legacy_answer_leaks,
)
from .scoring import score_submission
from .analysis import (
    build_sample_metadata,
    compare_pathways_detail,
    counts_dict_to_samples_by_genes,
    filter_counts_by_minimum_total,
    gseapy_available,
    load_counts_csv_as_samples_by_genes,
    load_author_de_table_csv,
    merge_analysis_options,
    enrichr_ora,
    ora_fisher,
    overlap_genes_across_top_pathways,
    pick_de_query_genes,
    pydeseq2_available,
    run_deseq2_contrast,
    top_hits_statistically_close,
    validate_counts_case,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_TRACE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "pathway_traces"


def load_case(
    case_name: str = "toy_case_001.json", *, agent_safe: bool = False
) -> Dict[str, Any]:
    """Load a case JSON. Set ``agent_safe=True`` to omit orchestrator secret fields."""
    case, _secrets = load_case_file(DATA_DIR, case_name, agent_safe=agent_safe)
    return case


def _legacy_de_rows(top_names: List[str]) -> List[Dict[str, Any]]:
    """Synthetic DE rows for legacy JSON-only cases."""
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(top_names):
        rows.append(
            {
                "gene": name,
                "baseMean": 500.0,
                "log2FoldChange": 2.0 - i * 0.1,
                "lfcSE": 0.2,
                "pvalue": 1e-6,
                "padj": 0.01,
                "significant": True,
            }
        )
    return rows


def _write_html_trace(
    episode_id: str,
    steps: List[Dict[str, Any]],
    case_id: str,
) -> str:
    OUTPUT_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_TRACE_DIR / f"{episode_id}.html"
    rows_html = []
    for s in steps:
        rows_html.append(
            "<tr><td>{}</td><td><pre>{}</pre></td><td>{}</td></tr>".format(
                html.escape(str(s.get("step", ""))),
                html.escape(json.dumps(s.get("detail", {}), indent=2)[:8000]),
                html.escape(str(s.get("message", ""))[:2000]),
            )
        )
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Pathway trace {html.escape(episode_id)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:1rem;}} table{{border-collapse:collapse;width:100%;}}
td,th{{border:1px solid #ccc;padding:0.4rem;vertical-align:top;}} pre{{white-space:pre-wrap;}}</style>
</head><body>
<h1>Pathway analysis episode</h1>
<p><b>case</b>: {html.escape(case_id)} &nbsp; <b>episode</b>: {html.escape(episode_id)}</p>
<p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p>
<table><thead><tr><th>Step</th><th>Detail</th><th>Message</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
</body></html>"""
    path.write_text(body, encoding="utf-8")
    return str(path)


def _safe_case_id(case: Dict[str, Any]) -> str:
    """Best-effort case identifier for trace rendering."""
    try:
        cid = case.get("case_id")
    except Exception:
        cid = None
    return str(cid or "unknown_case")


class PathwayEnvironment(Environment):
    """
    Pathway inference with optional **pipeline Mode A** (counts + metadata in JSON),
    or **legacy** toy fixtures (static gene/pathway lists).
    """

    def __init__(
        self,
        case_file: str = "toy_case_001.json",
        *,
        agent_safe_cases: bool = False,
    ):
        super().__init__()
        self._case_file = case_file
        self._agent_safe_cases = agent_safe_cases
        self._case: Dict[str, Any] = {}
        self._state = PathwayState()
        self._true_pathway: str = ""
        self._true_pathway_aliases: List[str] = []
        self._expected_keywords: List[str] = []
        self._eval_mode: bool = True
        self._orchestrator_mode: bool = False
        self._max_steps: int = 30
        self._episode_outcome: Optional[Dict[str, Any]] = None
        self._de_rows: List[Dict[str, Any]] = []
        self._ora_rows: List[Dict[str, Any]] = []
        self._query_genes: List[str] = []
        self._trace_steps: List[Dict[str, Any]] = []
        self._universe_genes: List[str] = []
        self.reset()

    def set_case_file(self, case_file: str) -> None:
        """Switch JSON case before ``reset()`` (used by the Gradio Pathway lab tab)."""
        self._case_file = case_file

    @property
    def episode_outcome(self) -> Optional[Dict[str, Any]]:
        """Orchestrator-only score after ``submit_answer`` (not exposed via agent state)."""
        return self._episode_outcome

    def _emit(self, obs: PathwayObservation) -> PathwayObservation:
        if obs.trace_path is None and self._state.episode_id:
            obs = obs.model_copy(update={"trace_path": self._refresh_trace_file()})
        return sanitize_observation_for_agent(
            obs,
            eval_mode=self._eval_mode,
            orchestrator_mode=self._orchestrator_mode,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PathwayObservation:
        use_agent_safe = bool(
            kwargs.get("agent_safe_cases", self._agent_safe_cases)
        )
        full_case, secrets = load_case_file(
            DATA_DIR, self._case_file, agent_safe=False
        )
        self._eval_mode = resolve_eval_mode(full_case, kwargs)
        self._orchestrator_mode = resolve_orchestrator_mode(full_case, kwargs)
        if use_agent_safe or (
            self._eval_mode and not self._orchestrator_mode
        ):
            self._case = strip_case_secrets(full_case)
        else:
            self._case = full_case
        eid = episode_id or str(uuid.uuid4())
        strict = bool(kwargs.get("strict", full_case.get("strict_mode", False)))
        self._max_steps = default_max_steps(full_case)
        self._true_pathway = str(secrets.get("true_pathway", ""))
        self._true_pathway_aliases = list(secrets.get("true_pathway_aliases") or [])
        self._expected_keywords = list(secrets.get("expected_keywords") or [])
        self._episode_outcome = None
        pipeline = (
            (
                "counts" in self._case
                or "counts_file" in self._case
                or "de_table_file" in self._case
            )
            and "sample_ids" in self._case
            and "sample_metadata" in self._case
        )
        self._de_rows = []
        self._ora_rows = []
        self._query_genes = []
        self._trace_steps = []
        self._universe_genes = []
        self._state = PathwayState(
            episode_id=eid,
            step_count=0,
            conditions=list(self._case.get("conditions", [])),
            pipeline_mode=pipeline,
            strict_mode=strict,
            legacy_mode=not pipeline,
            eval_mode=self._eval_mode,
            max_steps=self._max_steps,
        )
        mode = "legacy"
        if pipeline:
            if "de_table_file" in self._case:
                mode = "author_de_table"
            elif "counts_file" in self._case or "counts" in self._case:
                mode = "counts_matrix"
            else:
                mode = "pipeline_unknown"
        msg = (
            "Dataset loaded (pipeline: counts/metadata)."
            if mode == "counts_matrix"
            else (
                "Dataset loaded (pipeline: author DE table; enrichment only, not DESeq2-from-counts)."
                if mode == "author_de_table"
                else "Toy dataset loaded (legacy static lists)."
            )
        )
        self._trace(
            "reset",
            {
                "case_id": self._case.get("case_id"),
                "pipeline": pipeline,
                "mode": mode,
                "strict": strict,
            },
            msg,
        )
        trace_path = _write_html_trace(
            eid, self._trace_steps, _safe_case_id(self._case)
        )
        obs = PathwayObservation(
            message=msg
            + " Use understand_experiment_design, inspect, run DE, enrichment, compare, or submit.",
            available_conditions=self._state.conditions,
            metadata={
                "case_id": self._case["case_id"],
                "pipeline_mode": pipeline,
                "pipeline_data_mode": mode,
                "eval_mode": self._eval_mode,
                "max_steps": self._max_steps,
            },
            trace_path=trace_path,
        )
        return self._emit(obs)

    def _trace(self, kind: str, detail: Dict[str, Any], message: str) -> None:
        s = self._state
        self._trace_steps.append(
            {
                "step": s.step_count,
                "kind": kind,
                "detail": detail,
                "message": message,
            }
        )

    def _refresh_trace_file(self) -> str:
        eid = self._state.episode_id or "unknown"
        return _write_html_trace(
            eid, self._trace_steps, _safe_case_id(self._case)
        )

    def _fail_strict(
        self, reason: str, failure_code: str = FC.STRICT_TERMINATION
    ) -> PathwayObservation:
        self._state.is_done = True
        self._trace(
            "strict_failure",
            {"reason": reason, "failure_code": failure_code},
            reason,
        )
        tp = self._refresh_trace_file()
        return PathwayObservation(
            message=reason,
            done=True,
            reward=-3.0,
            metadata={
                "strict_failure": True,
                "reason": reason,
                "failure_code": failure_code,
            },
            trace_path=tp,
        )

    def step(
        self,
        action: PathwayAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PathwayObservation:
        return self._emit(self._step_inner(action))

    def _step_inner(self, action: PathwayAction) -> PathwayObservation:
        s = self._state

        if s.is_done:
            obs = PathwayObservation(
                message="Episode already finished; call reset() for a new episode.",
                done=True,
                reward=0.0,
                metadata={
                    "error": "episode_done",
                    "failure_code": FC.EPISODE_ALREADY_DONE,
                    "step_count": s.step_count,
                },
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        s.step_count += 1

        if self._eval_mode and s.step_count > self._max_steps:
            s.is_done = True
            self._trace(
                "max_steps",
                {"max_steps": self._max_steps},
                "Step budget exhausted.",
            )
            return PathwayObservation(
                message="Maximum steps exceeded for this episode.",
                done=True,
                reward=-1.0 if not self._eval_mode else 0.0,
                metadata={
                    "failure_code": FC.MAX_STEPS_EXCEEDED,
                    "max_steps": self._max_steps,
                },
                trace_path=self._refresh_trace_file(),
            )

        if action.action_type == "inspect_dataset":
            meta = self._case.get("sample_metadata") or {}
            sample_ids = list(self._case.get("sample_ids") or [])
            sample_level = bool(sample_ids and meta)
            if s.legacy_mode:
                msg = (
                    "Legacy fixture: conditions are listed; per-sample metadata and counts "
                    "are not modeled. DE and ORA return static curated outputs."
                )
            elif sample_level:
                msg = (
                    "Sample metadata and conditions are available for contrast specification."
                )
            else:
                msg = (
                    "Conditions are available; sample_ids or sample_metadata are incomplete "
                    "in this case."
                )
            inspect_meta: Dict[str, Any] = {
                "step_count": s.step_count,
                "legacy_mode": s.legacy_mode,
                "pipeline_mode": s.pipeline_mode,
                "sample_level_metadata_available": sample_level,
                "sample_metadata": meta,
                "sample_ids": sample_ids,
                "pydeseq2_available": pydeseq2_available(),
                "experiment_metadata": self._case.get("experiment_metadata"),
            }
            if s.legacy_mode and not self._eval_mode:
                inspect_meta["static_top_genes"] = list(
                    self._case.get("top_genes") or []
                )
                inspect_meta["static_top_pathways"] = list(
                    self._case.get("top_pathways") or []
                )
            inspect_meta = strip_legacy_answer_leaks(
                inspect_meta, eval_mode=self._eval_mode
            )
            obs = PathwayObservation(
                message=msg,
                available_conditions=s.conditions,
                reward=shaping_reward(self._eval_mode, 0.05),
                metadata=inspect_meta,
            )
            self._trace(
                "inspect_dataset",
                {"conditions": s.conditions, "legacy": s.legacy_mode},
                obs.message,
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        if action.action_type == "understand_experiment_design":
            return self._step_understand_experiment_design(action)

        if action.action_type == "run_differential_expression":
            return self._step_de(action)

        if action.action_type == "run_pathway_enrichment":
            return self._step_enrichment(action)

        if action.action_type == "compare_pathways":
            return self._step_compare(action)

        if action.action_type == "submit_answer":
            return self._step_submit(action)

        obs = PathwayObservation(
            message=f"Unknown action_type: {action.action_type}",
            reward=shaping_reward(self._eval_mode, -0.2),
            metadata={
                "step_count": s.step_count,
                "failure_code": FC.UNKNOWN_ACTION_TYPE,
                "action_type": action.action_type,
            },
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    def _experiment_design_dict(self) -> Dict[str, Any]:
        case = self._case
        s = self._state
        sample_ids = list(case.get("sample_ids") or [])
        smd = case.get("sample_metadata") or {}
        per: Dict[str, int] = {}
        for sid in sample_ids:
            c = smd.get(sid)
            if c is not None:
                per[c] = per.get(c, 0) + 1
        conds = list(s.conditions)
        sample_level = bool(sample_ids and smd)
        design: Dict[str, Any] = {
            "case_id": case.get("case_id"),
            "pipeline_mode": s.pipeline_mode,
            "legacy_mode": s.legacy_mode,
            "conditions": conds,
            "n_groups": len(conds),
            "n_samples": len(sample_ids) if sample_level else None,
            "sample_ids": sample_ids,
            "sample_level_metadata_available": sample_level,
            "default_contrast": case.get("default_contrast"),
            "experiment_metadata": case.get("experiment_metadata"),
        }
        if sample_level:
            design["samples_per_condition"] = per
            workflow = (
                "(1) Groups: use conditions + samples_per_condition to see how many groups and "
                "replicates exist. (2) DGE: pick reference vs alternate for DESeq2 "
                "(validate via understand_experiment_design or pass to run_differential_expression). "
                "(3) Pathways: run_pathway_enrichment then compare/submit."
            )
            note = (
                "Reference = baseline (denominator of log2 fold change); alternate = comparison arm "
                "for DGE. Optionally set condition_a / condition_b here to validate before "
                "run_differential_expression."
            )
        elif s.legacy_mode:
            design["samples_per_condition"] = None
            design["legacy_fixture"] = True
            workflow = (
                "(1) Groups: conditions are named only (no per-sample counts in this legacy fixture). "
                "(2) DGE / ORA return static curated gene and pathway lists. "
                "(3) Submit the pathway hypothesis."
            )
            note = (
                "Legacy mode does not run DESeq2 on counts. Contrast validation checks condition "
                "names only. Use run_differential_expression and run_pathway_enrichment for "
                "fixture outputs, then submit_answer."
            )
        else:
            design["samples_per_condition"] = per if per else None
            workflow = (
                "(1) Groups: conditions are listed; sample-level metadata may be incomplete. "
                "(2) DGE: pick reference vs alternate when counts/metadata are available. "
                "(3) Pathways: enrichment then submit."
            )
            note = (
                "Reference = baseline; alternate = comparison arm. Sample counts per condition "
                "are unavailable until sample_ids and sample_metadata are present in the case."
            )
        design["agent_workflow"] = workflow
        design["design_note"] = note
        return design

    def _validate_contrast_proposal(
        self, ref: str, alt: str
    ) -> Optional[Tuple[str, str]]:
        """Return (error_message, failure_code) if invalid; None if valid for DESeq2."""
        conds = set(self._state.conditions)
        if ref not in conds or alt not in conds:
            return (
                "Reference and alternate must be among the case `conditions`.",
                FC.DESIGN_INVALID_CONTRAST_NAMES,
            )
        if ref == alt:
            return (
                "Reference and alternate must be two different conditions.",
                FC.DESIGN_INVALID_CONTRAST_NAMES,
            )
        sample_ids = list(self._case.get("sample_ids") or [])
        smd = self._case.get("sample_metadata") or {}
        if not sample_ids:
            return None
        per: Dict[str, int] = {}
        for sid in sample_ids:
            c = smd.get(sid)
            if c is not None:
                per[c] = per.get(c, 0) + 1
        if per.get(ref, 0) < 1 or per.get(alt, 0) < 1:
            return (
                "Each contrast arm must have at least one sample in `sample_metadata`.",
                FC.DESIGN_INSUFFICIENT_SAMPLES_PER_ARM,
            )
        return None

    def _step_understand_experiment_design(
        self, action: PathwayAction
    ) -> PathwayObservation:
        s = self._state
        design = self._experiment_design_dict()
        ref_in = (action.condition_a or "").strip()
        alt_in = (action.condition_b or "").strip()
        has_both = bool(ref_in and alt_in)
        has_partial = bool(ref_in or alt_in) and not has_both

        if has_partial:
            obs = PathwayObservation(
                message=(
                    "Provide both reference (condition_a) and alternate (condition_b) to "
                    "validate a contrast, or leave both empty for a design summary only."
                ),
                available_conditions=s.conditions,
                experiment_design=design,
                reward=shaping_reward(self._eval_mode, -0.02),
                metadata={
                    "step_count": s.step_count,
                    "validation": "incomplete",
                    "failure_code": FC.DESIGN_PARTIAL_CONTRAST,
                },
            )
            self._trace(
                "understand_experiment_design",
                {"validation": "incomplete"},
                obs.message,
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        if not has_both:
            s.design_understood = True
            if s.legacy_mode:
                msg = (
                    "Design summary (legacy fixture): condition names are available; per-sample "
                    "replicate counts are not modeled. DE and ORA use static outputs. You may still "
                    "validate a contrast by naming reference vs alternate, then run DE → ORA → submit."
                )
            elif design.get("sample_level_metadata_available"):
                msg = (
                    "Design summary: you have the groups (conditions) and sample counts per group. "
                    "Next, choose reference vs alternate for DGE (differential expression), then "
                    "pathway steps. Re-run this action with both conditions set to validate your "
                    "contrast."
                )
            else:
                msg = (
                    "Design summary: condition names are listed; sample counts per group are not "
                    "available in this case. Re-run with both conditions set to validate a contrast "
                    "when supported, then run DGE and pathway steps."
                )
            obs = PathwayObservation(
                message=msg,
                available_conditions=s.conditions,
                experiment_design=design,
                reward=shaping_reward(self._eval_mode, 0.05),
                metadata={"step_count": s.step_count, "validation": "summary_only"},
            )
            self._trace("understand_experiment_design", {"mode": "summary"}, msg)
            obs.trace_path = self._refresh_trace_file()
            return obs

        invalid = self._validate_contrast_proposal(ref_in, alt_in)
        if invalid:
            err, fcode = invalid
            s.validated_reference = None
            s.validated_alternate = None
            s.design_understood = True
            obs = PathwayObservation(
                message=err,
                available_conditions=s.conditions,
                experiment_design=design,
                reward=shaping_reward(self._eval_mode, -0.05),
                metadata={
                    "step_count": s.step_count,
                    "validation": "invalid",
                    "failure_code": fcode,
                },
            )
            self._trace(
                "understand_experiment_design",
                {"validation": "invalid", "proposal": [ref_in, alt_in]},
                err,
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        s.validated_reference = ref_in
        s.validated_alternate = alt_in
        s.design_understood = True
        design["validated_contrast"] = {"reference": ref_in, "alternate": alt_in}
        msg = (
            f"DGE contrast chosen: reference=`{ref_in}`, alternate=`{alt_in}` "
            f"({len(s.conditions)} groups in study). "
            "run_differential_expression will use this pair when DE omits conditions; "
            "explicit DE fields override. Then run pathway enrichment."
        )
        obs = PathwayObservation(
            message=msg,
            available_conditions=s.conditions,
            experiment_design=design,
            reward=shaping_reward(self._eval_mode, 0.08),
            metadata={"step_count": s.step_count, "validation": "valid"},
        )
        self._trace(
            "understand_experiment_design",
            {"validation": "valid", "contrast": [ref_in, alt_in]},
            msg,
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    def _resolve_de_contrast(
        self, action: PathwayAction
    ) -> tuple[Optional[str], Optional[str]]:
        """DESeq2 contrast: explicit action fields beat validated design, then default_contrast."""
        dc = self._case.get("default_contrast") or {}
        ar = (action.condition_a or "").strip()
        ab = (action.condition_b or "").strip()
        ref = ar or self._state.validated_reference or dc.get("reference")
        alt = ab or self._state.validated_alternate or dc.get("alternate")
        return ref, alt

    def _step_de(self, action: PathwayAction) -> PathwayObservation:
        s = self._state
        if s.legacy_mode:
            names = list(self._case.get("top_genes", []))
            self._de_rows = _legacy_de_rows(names)
            self._query_genes = names
            s.de_run = True
            self._trace("de", {"legacy": True, "genes": names}, "Legacy DE")
            obs = PathwayObservation(
                message="Differential expression complete (legacy fixture).",
                top_genes=names,
                de_genes=self._de_rows,
                reward=shaping_reward(self._eval_mode, 0.25),
                metadata={"step_count": s.step_count, "legacy": True},
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        if not pydeseq2_available():
            if s.strict_mode:
                return self._fail_strict(
                    "PyDESeq2 is not installed; strict mode terminates.",
                    FC.DE_PYDESeq2_UNAVAILABLE,
                )
            return PathwayObservation(
                message="PyDESeq2 is not installed; cannot run DE on counts.",
                reward=shaping_reward(self._eval_mode, -0.5),
                metadata={
                    "error": "missing_pydeseq2",
                    "failure_code": FC.DE_PYDESeq2_UNAVAILABLE,
                },
            )

        ref, alt = self._resolve_de_contrast(action)
        if not ref or not alt:
            msg = "Specify condition_a (reference) and condition_b (alternate) for DESeq2."
            if s.strict_mode:
                return self._fail_strict(msg, FC.DE_MISSING_CONTRAST)
            return PathwayObservation(
                message=msg,
                reward=shaping_reward(self._eval_mode, -0.3),
                metadata={"error": "contrast", "failure_code": FC.DE_MISSING_CONTRAST},
            )

        sample_ids = self._case["sample_ids"]
        smd = self._case["sample_metadata"]
        try:
            if "de_table_file" in self._case:
                # Author-provided DE (no counts available). We treat this as a precomputed DE run.
                opts = merge_analysis_options(self._case)
                de_rows = load_author_de_table_csv(
                    DATA_DIR / str(self._case["de_table_file"]),
                    gene_column=str(self._case.get("de_table_gene_column") or "Gene,name"),
                    log2fc_column=str(self._case.get("de_table_log2fc_column") or "log2FoldChange"),
                    pvalue_column=str(self._case.get("de_table_pvalue_column") or "pvalue"),
                    padj_column=str(self._case.get("de_table_padj_column") or "padj"),
                )
                padj_alpha = float(opts["padj_alpha"])
                for r in de_rows:
                    try:
                        pv = float(r.get("padj"))
                    except (TypeError, ValueError):
                        pv = 1.0
                    r["significant"] = bool(pv <= padj_alpha)
                self._de_rows = de_rows
                self._query_genes = pick_de_query_genes(
                    de_rows,
                    padj_alpha=padj_alpha,
                    direction=str(opts["de_query_direction"]),
                    min_abs_log2fc=float(opts["min_abs_log2fc"]),
                )
                self._universe_genes = []  # unknown without counts
                s.de_run = True
                top_names = [r["gene"] for r in de_rows[:50]]
                self._trace(
                    "de",
                    {
                        "precomputed": True,
                        "source": "author_de_table",
                        "contrast": [ref, alt],
                        "n_sig": sum(1 for r in de_rows if r.get("significant")),
                        "n_rows": len(de_rows),
                    },
                    "Differential expression loaded (author-provided table).",
                )
                obs = PathwayObservation(
                    message="Differential expression loaded from author table.",
                    top_genes=top_names,
                    de_genes=self._de_rows,
                    reward=shaping_reward(self._eval_mode, 0.25),
                    metadata={
                        "step_count": s.step_count,
                        "precomputed": True,
                        "source": "author_de_table",
                    },
                )
                obs.trace_path = self._refresh_trace_file()
                return obs

            if "counts_file" in self._case:
                counts_df = load_counts_csv_as_samples_by_genes(
                    DATA_DIR / str(self._case["counts_file"]),
                    sample_ids=sample_ids,
                )
            else:
                counts = self._case["counts"]
                v_err = validate_counts_case(self._case)
                if v_err:
                    raise ValueError(v_err)
                counts_df = counts_dict_to_samples_by_genes(counts, sample_ids)
            meta_df = build_sample_metadata(sample_ids, smd)
        except ValueError as exc:
            if s.strict_mode:
                return self._fail_strict(str(exc), FC.DE_INVALID_COUNTS_MATRIX)
            return PathwayObservation(
                message=str(exc),
                reward=shaping_reward(self._eval_mode, -0.5),
                metadata={
                    "error": "counts_or_metadata_invalid",
                    "failure_code": FC.DE_INVALID_COUNTS_MATRIX,
                },
            )

        opts = merge_analysis_options(self._case)
        counts_df, n_genes_in, n_genes_filt = filter_counts_by_minimum_total(
            counts_df, int(opts["min_total_count"])
        )
        if n_genes_filt < 5:
            msg = (
                f"After min_total_count={opts['min_total_count']} prefilter, "
                f"only {n_genes_filt} genes remain (need ≥5 for stable DESeq2)."
            )
            if s.strict_mode:
                return self._fail_strict(msg, FC.DE_TOO_FEW_GENES_AFTER_FILTER)
            return PathwayObservation(
                message=msg,
                reward=shaping_reward(self._eval_mode, -0.5),
                metadata={
                    "error": "too_few_genes_after_filter",
                    "failure_code": FC.DE_TOO_FEW_GENES_AFTER_FILTER,
                },
            )

        rows, err = run_deseq2_contrast(
            counts_df,
            meta_df,
            alt,
            ref,
            padj_alpha=float(opts["padj_alpha"]),
        )
        if err:
            if s.strict_mode:
                return self._fail_strict(err, FC.DE_DESEQ2_FAILED)
            return PathwayObservation(
                message=err,
                reward=shaping_reward(self._eval_mode, -0.5),
                metadata={"error": err, "failure_code": FC.DE_DESEQ2_FAILED},
            )

        self._universe_genes = list(counts_df.columns)
        self._de_rows = rows
        self._query_genes = pick_de_query_genes(
            rows,
            padj_alpha=float(opts["padj_alpha"]),
            direction=str(opts["de_query_direction"]),
            min_abs_log2fc=float(opts["min_abs_log2fc"]),
        )
        s.de_run = True
        top_names = [r["gene"] for r in rows[:50]]
        self._trace(
            "de",
            {
                "contrast": [ref, alt],
                "n_sig": sum(1 for r in rows if r["significant"]),
                "genes_in_matrix": n_genes_in,
                "genes_after_prefilter": n_genes_filt,
            },
            "DESeq2 complete",
        )
        obs = PathwayObservation(
            message="Differential expression complete (PyDESeq2).",
            top_genes=top_names,
            de_genes=rows[:200],
            reward=shaping_reward(self._eval_mode, 0.35),
            metadata={
                "step_count": s.step_count,
                "contrast": [ref, alt],
                "genes_in_matrix": n_genes_in,
                "genes_after_prefilter": n_genes_filt,
                "analysis_options": {
                    k: opts[k]
                    for k in (
                        "min_total_count",
                        "padj_alpha",
                        "de_query_direction",
                        "min_abs_log2fc",
                    )
                },
            },
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    def _step_enrichment(self, action: PathwayAction) -> PathwayObservation:
        s = self._state
        if self._eval_mode and action.gene_list:
            return PathwayObservation(
                message=(
                    "Custom gene_list is disabled in eval mode; run differential "
                    "expression and use the resulting DE gene set for ORA."
                ),
                reward=shaping_reward(self._eval_mode, -0.2),
                metadata={"failure_code": FC.ORA_GENE_LIST_BLOCKED},
            )
        if not self._de_rows and not s.legacy_mode:
            msg = "Run differential expression before enrichment."
            return PathwayObservation(
                message=msg,
                reward=shaping_reward(self._eval_mode, -0.2),
                metadata={"failure_code": FC.ORA_DE_PREREQUISITE},
            )

        pathways = self._case.get("pathway_genes") or {}
        if s.legacy_mode:
            names = list(self._case.get("top_pathways", []))
            s.enrichment_run = True
            fake = [
                {
                    "pathway": n,
                    "p_value": 0.001,
                    "q_value": 0.01,
                    "overlap_genes": list(self._case.get("top_genes", []))[:2],
                    "overlap_count": 2,
                    "pathway_size": 10,
                    "de_in_universe": len(self._query_genes),
                    "gene_ratio": "2/10",
                }
                for n in names
            ]
            self._ora_rows = fake
            amb = top_hits_statistically_close(fake)
            ov = overlap_genes_across_top_pathways(fake)
            self._trace("ora", {"legacy": True}, "Legacy ORA")
            obs = PathwayObservation(
                message="Pathway enrichment complete (legacy fixture).",
                top_pathways=names,
                pathway_enrichment=fake,
                statistical_ambiguity=amb,
                overlap_summary=ov,
                reward=shaping_reward(self._eval_mode, 0.45),
                metadata={"legacy": True},
            )
            obs.trace_path = self._refresh_trace_file()
            return obs

        opts = merge_analysis_options(self._case)
        universe = self._universe_genes
        if not universe:
            if "counts" in self._case:
                universe = list(self._case["counts"].keys())
            else:
                universe = []
        query = action.gene_list if action.gene_list else self._query_genes
        if not query:
            query = pick_de_query_genes(
                self._de_rows,
                padj_alpha=float(opts["padj_alpha"]),
                direction=str(opts["de_query_direction"]),
                min_abs_log2fc=float(opts["min_abs_log2fc"]),
            )
        if not query and self._de_rows:
            query = [r["gene"] for r in self._de_rows[:50]]

        enrichr_libs = self._case.get("enrichr_libraries")
        if enrichr_libs:
            if not gseapy_available():
                msg = "gseapy not installed; cannot run Enrichr enrichment."
                if s.strict_mode:
                    return self._fail_strict(msg, FC.ORA_NO_PATHWAY_DEFINITIONS)
                return PathwayObservation(
                    message=msg,
                    reward=shaping_reward(self._eval_mode, -0.3),
                    metadata={
                        "error": "missing_gseapy",
                        "failure_code": FC.ORA_NO_PATHWAY_DEFINITIONS,
                    },
                )
            ora, err = enrichr_ora(
                query,
                libraries=list(enrichr_libs),
                background=universe or None,
                top_k=100,
            )
            if err:
                if s.strict_mode:
                    return self._fail_strict(err, FC.ORA_NO_PATHWAY_DEFINITIONS)
                return PathwayObservation(
                    message=err,
                    reward=shaping_reward(self._eval_mode, -0.3),
                    metadata={
                        "error": "enrichr_failed",
                        "failure_code": FC.ORA_NO_PATHWAY_DEFINITIONS,
                    },
                )
        else:
            if not pathways:
                msg = "Case has no pathway_genes (and no enrichr_libraries); cannot run ORA."
                if s.strict_mode:
                    return self._fail_strict(msg, FC.ORA_NO_PATHWAY_DEFINITIONS)
                return PathwayObservation(
                    message=msg,
                    reward=shaping_reward(self._eval_mode, -0.3),
                    metadata={
                        "error": "no_pathways",
                        "failure_code": FC.ORA_NO_PATHWAY_DEFINITIONS,
                    },
                )
            ora = ora_fisher(
                query,
                pathways,
                universe,
                min_pathway_genes=int(opts["ora_min_pathway_genes"]),
            )
        self._ora_rows = ora
        s.enrichment_run = True
        top_names = [r["pathway"] for r in ora[:20]]
        amb = top_hits_statistically_close(ora)
        ov = overlap_genes_across_top_pathways(ora)
        self._trace("ora", {"n_pathways": len(ora)}, "ORA complete")
        obs = PathwayObservation(
            message="Over-representation analysis complete.",
            top_pathways=top_names,
            pathway_enrichment=ora[:50],
            statistical_ambiguity=amb,
            overlap_summary=ov,
            reward=shaping_reward(self._eval_mode, 0.5),
            metadata={
                "query_genes": len(query),
                "ora_universe_size": len(universe),
                "ora_min_pathway_genes": int(opts["ora_min_pathway_genes"]),
            },
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    def _step_compare(self, action: PathwayAction) -> PathwayObservation:
        s = self._state
        if self._eval_mode and not s.enrichment_run:
            return PathwayObservation(
                message="Run pathway enrichment before compare_pathways.",
                reward=shaping_reward(self._eval_mode, -0.1),
                metadata={"failure_code": FC.COMPARE_REQUIRES_ORA},
            )
        a = (action.pathway_a or "").strip()
        b = (action.pathway_b or "").strip()
        if not a or not b:
            return PathwayObservation(
                message="Provide pathway_a and pathway_b.",
                reward=shaping_reward(self._eval_mode, -0.1),
                metadata={
                    "error": "missing_names",
                    "failure_code": FC.COMPARE_MISSING_PATHWAY_NAMES,
                },
            )
        pathways = self._case.get("pathway_genes") or {}
        if s.legacy_mode:
            # infer dummy pathways from top_pathways list
            pathways = {
                p: self._case.get("top_genes", [])
                for p in self._case.get("top_pathways", [])
            }
        detail = compare_pathways_detail(
            a, b, pathways, self._query_genes or list(self._case.get("top_genes", []))
        )
        self._trace("compare_pathways", detail, f"Compared {a} vs {b}")
        obs = PathwayObservation(
            message=f"Pathway comparison: {a} vs {b}.",
            pathway_comparison=detail,
            reward=shaping_reward(self._eval_mode, 0.15),
            metadata={"step_count": s.step_count},
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    def _step_submit(self, action: PathwayAction) -> PathwayObservation:
        s = self._state
        hypothesis = (action.hypothesis or "").strip()
        if not hypothesis:
            return PathwayObservation(
                message="Provide a non-empty pathway hypothesis.",
                reward=shaping_reward(self._eval_mode, -0.1),
                metadata={"failure_code": FC.SUBMIT_EMPTY_HYPOTHESIS},
            )
        if self._eval_mode:
            if not s.de_run:
                return PathwayObservation(
                    message="Run differential expression before submitting.",
                    reward=0.0,
                    metadata={"failure_code": FC.SUBMIT_PREREQUISITE_DE},
                )
            if not s.enrichment_run:
                return PathwayObservation(
                    message="Run pathway enrichment before submitting.",
                    reward=0.0,
                    metadata={"failure_code": FC.SUBMIT_PREREQUISITE_ORA},
                )

        top_ora = [r.get("pathway", "") for r in self._ora_rows[:20] if r.get("pathway")]
        outcome = score_submission(
            hypothesis,
            true_pathway=self._true_pathway,
            expected_keywords=self._expected_keywords,
            pathway_gene_set_names=list((self._case.get("pathway_genes") or {}).keys()),
            true_pathway_aliases=self._true_pathway_aliases,
            top_ora_pathways=top_ora,
        )
        correct = bool(outcome.get("correct"))
        self._episode_outcome = {
            **outcome,
            "hypothesis": hypothesis,
            "step_count": s.step_count,
            "case_id": self._case.get("case_id"),
        }
        s.is_done = True
        self._trace(
            "submit",
            {
                "hypothesis": hypothesis,
                "correct": correct,
                "match_mode": outcome.get("match_mode"),
            },
            "Episode end",
        )
        meta: Dict[str, Any] = {
            "correct": correct,
            "episode_score": outcome,
            "step_count": s.step_count,
        }
        if not correct:
            meta["failure_code"] = FC.SUBMIT_INCORRECT_HYPOTHESIS
        nominal_reward = 2.0 if correct else -1.0
        obs = PathwayObservation(
            message=(
                "Answer submitted. Episode complete."
                if self._eval_mode
                else ("Correct pathway." if correct else "Incorrect pathway.")
            ),
            done=True,
            reward=shaping_reward(self._eval_mode, nominal_reward)
            if not self._eval_mode
            else 0.0,
            metadata=meta,
        )
        obs.trace_path = self._refresh_trace_file()
        return obs

    @property
    def state(self) -> PathwayState:
        return self._state
