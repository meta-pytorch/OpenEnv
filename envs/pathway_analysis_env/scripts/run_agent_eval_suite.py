#!/usr/bin/env python3
"""
Run the standard pathway agent eval manifest (fixed policy baseline).

Scores via ``env.episode_outcome`` (orchestrator mode). Writes JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.analysis import gseapy_available, pydeseq2_available
from pathway_analysis_env.server.pathway_environment import (
    DATA_DIR,
    PathwayEnvironment,
)


def _run_episode(spec: Dict[str, Any], *, strict: bool) -> Dict[str, Any]:
    case_file = spec["case_file"]
    if spec.get("requires_pydeseq2") and not pydeseq2_available():
        return {
            "id": spec["id"],
            "skipped": True,
            "reason": "pydeseq2_unavailable",
        }
    if spec.get("requires_gseapy") and not gseapy_available():
        return {
            "id": spec["id"],
            "skipped": True,
            "reason": "gseapy_unavailable",
        }

    from pathway_analysis_env.server.pathway_environment import load_case

    case = load_case(case_file)
    ref = (case.get("default_contrast") or {}).get("reference")
    alt = (case.get("default_contrast") or {}).get("alternate")

    env = PathwayEnvironment(case_file=case_file)
    env.reset(strict=strict, orchestrator_mode=True)

    actions: List[str] = []

    def go(kind: str, **kw: Any):
        actions.append(kind)
        return env.step(PathwayAction(action_type=kind, **kw))

    go("understand_experiment_design")
    go("inspect_dataset")
    o_de = go(
        "run_differential_expression",
        condition_a=ref,
        condition_b=alt,
    )
    if o_de.metadata and o_de.metadata.get("failure_code"):
        return {
            "id": spec["id"],
            "case_file": case_file,
            "passed": False,
            "stage": "de",
            "failure_code": o_de.metadata.get("failure_code"),
            "actions": actions,
        }
    o_ora = go("run_pathway_enrichment")
    if o_ora.metadata and o_ora.metadata.get("failure_code"):
        return {
            "id": spec["id"],
            "case_file": case_file,
            "passed": False,
            "stage": "ora",
            "failure_code": o_ora.metadata.get("failure_code"),
            "actions": actions,
        }
    hyp = spec.get("hypothesis", "")
    go("submit_answer", hypothesis=hyp)
    outcome = env.episode_outcome or {}
    return {
        "id": spec["id"],
        "case_file": case_file,
        "passed": bool(outcome.get("correct")),
        "episode_outcome": outcome,
        "actions": actions,
        "steps": env.state.step_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIR / "eval_manifest.json",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [_run_episode(ep, strict=args.strict) for ep in manifest.get("episodes", [])]
    n_pass = sum(1 for r in results if r.get("passed"))
    n_run = sum(1 for r in results if not r.get("skipped"))
    summary = {
        "manifest": str(args.manifest),
        "passed": n_pass,
        "run": n_run,
        "total": len(results),
        "pass_rate": (n_pass / n_run) if n_run else 0.0,
        "results": results,
    }
    text = json.dumps(summary, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
