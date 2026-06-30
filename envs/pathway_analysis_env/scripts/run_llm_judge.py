#!/usr/bin/env python3
"""
LLM-judge evaluation for pathway_analysis_env (eval-only, non-deterministic).

Runs a tool-calling agent on one GEO case, asks it to produce a findings
report, builds a *reference* report from the same live episode outputs (DE/ORA)
that the agent saw, then asks a judge model to rate the agent report.

This is an EVALUATION aid only. It is deliberately NOT wired into the
environment reward (which must stay deterministic for RL training).

Usage:
  export OPENAI_API_KEY=...
  PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_judge.py \
    --case geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json \
    --reference-dir envs/pathway_analysis_env/data/geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso \
    --agent-model gpt-4o-mini --judge-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI

from pathway_analysis_env.agent_openai_tools import (
    OPENAI_TOOLS,
    observation_to_tool_result_content,
    tool_call_to_pathway_action,
)
from pathway_analysis_env.server.pathway_environment import DATA_DIR, PathwayEnvironment

AGENT_SYSTEM_PROMPT = """You are a computational biologist agent operating a pathway analysis environment.

Workflow: understand_experiment_design / inspect_dataset -> run_differential_expression
(reference vs alternate) -> run_pathway_enrichment -> optionally compare_pathways ->
submit_answer with the activated pathway. Never guess before running DE and ORA.
After you submit, you will be asked to write a short findings report."""

REPORT_REQUEST = """The episode is complete. Write a concise findings report (5-8 sentences)
of what you discovered: the contrast you ran, the most significant differentially
expressed genes/direction, the top enriched pathways, and your biological interpretation
of what program is activated/repressed in this experiment."""


def _chat_kwargs_for_model(model: str) -> Dict[str, Any]:
    """
    Some models (e.g. gpt-5) do not accept non-default temperature values.
    Return a safe kwargs dict for chat.completions.create.
    """
    if model.startswith("gpt-5"):
        return {}
    return {"temperature": 0.0}


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def build_reference_report_from_live(
    case: Dict[str, Any],
    *,
    de_rows: List[Dict[str, Any]],
    ora_rows: List[Dict[str, Any]],
    contrast: str,
) -> str:
    """Build reference report from the same live outputs seen by the agent."""
    top = []
    for row in ora_rows[:10]:
        name = row.get("pathway")
        q = row.get("q_value")
        q_txt = f"{q:.2e}" if isinstance(q, (int, float)) else str(q)
        genes = ", ".join((row.get("overlap_genes") or [])[:8])
        top.append(f"  - {name} (q={q_txt}); key genes: {genes}")
    top_block = "\n".join(top) if top else "  - (none)"

    sig_n = sum(1 for r in de_rows if bool(r.get("significant")))
    meta = case.get("experiment_metadata", {})
    return (
        f"STUDY: {meta.get('accession')} - {meta.get('summary')}\n"
        f"CONTRAST: {contrast}\n"
        f"SIGNIFICANT GENES (padj<0.05): {sig_n}\n"
        f"TOP ENRICHED PATHWAYS (ground-truth, from same live episode outputs):\n"
        f"{top_block}\n"
    )


async def run_agent_report(
    client: AsyncOpenAI, model: str, case_file: str
) -> tuple[str, str]:
    env = PathwayEnvironment(case_file=case_file)
    obs = env.reset(orchestrator_mode=True)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Case: {case_file}. Conditions: {obs.available_conditions}. {obs.message}",
        },
    ]
    for _ in range(20):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            **_chat_kwargs_for_model(model),
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            if env.state.is_done:
                break
            continue
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            action = tool_call_to_pathway_action(
                name=tc.function.name, arguments_json=tc.function.arguments)
            step_obs = env.step(action)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": observation_to_tool_result_content(step_obs),
            })
        if env.state.is_done:
            break

    # Build reference from exactly the outputs this episode produced.
    de_rows = list(getattr(env, "_de_rows", []) or [])
    ora_rows = list(getattr(env, "_ora_rows", []) or [])
    c_ref = getattr(env._state, "validated_reference", None) or (
        (env._case.get("default_contrast") or {}).get("reference")
    )
    c_alt = getattr(env._state, "validated_alternate", None) or (
        (env._case.get("default_contrast") or {}).get("alternate")
    )
    contrast = (
        f"{c_alt} vs {c_ref} (reference={c_ref})" if c_ref and c_alt else "unknown"
    )
    reference = build_reference_report_from_live(
        env._case, de_rows=de_rows, ora_rows=ora_rows, contrast=contrast
    )

    messages.append({"role": "user", "content": REPORT_REQUEST})
    resp = await client.chat.completions.create(
        model=model, messages=messages, **_chat_kwargs_for_model(model)
    )
    return resp.choices[0].message.content or "", reference


JUDGE_SYSTEM = """You are a strict scientific reviewer. Compare an AGENT REPORT against a
REFERENCE (ground-truth pathway-analysis result). Score how well the agent recovered the
correct biology. Return STRICT JSON only."""

JUDGE_RUBRIC = """Score 0.0-1.0 on each criterion, then an overall 0.0-1.0:
- primary_biology: did the agent identify the correct dominant program?
- supporting_pathways: did it mention the secondary/related pathways?
- evidence_grounding: are claims tied to the actual DE/enrichment results (not generic priors)?
- mechanism: correct biological interpretation of the experiment?
Return JSON: {"primary_biology":x,"supporting_pathways":x,"evidence_grounding":x,
"mechanism":x,"overall":x,"justification":"2-3 sentences"}"""


async def judge(client: AsyncOpenAI, model: str, agent_report: str, reference: str) -> Dict[str, Any]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"{JUDGE_RUBRIC}\n\n=== REFERENCE ===\n{reference}\n\n=== AGENT REPORT ===\n{agent_report}",
            },
        ],
        response_format={"type": "json_object"},
        **_chat_kwargs_for_model(model),
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def main_async(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    case_path = DATA_DIR / args.case
    case = json.loads(case_path.read_text())

    print("Running agent and generating report...", flush=True)
    agent_report, reference = await run_agent_report(client, args.agent_model, args.case)
    print("Judging...", flush=True)
    verdict = await judge(client, args.judge_model, agent_report, reference)

    print("\n" + "=" * 70)
    print("REFERENCE REPORT\n" + "-" * 70)
    print(reference)
    print("=" * 70)
    print(f"AGENT REPORT ({args.agent_model})\n" + "-" * 70)
    print(agent_report)
    print("=" * 70)
    print(f"JUDGE VERDICT ({args.judge_model})\n" + "-" * 70)
    print(json.dumps(verdict, indent=2))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "case": args.case,
                    "agent_model": args.agent_model,
                    "judge_model": args.judge_model,
                    "reference": reference,
                    "agent_report": agent_report,
                    "verdict": verdict,
                    "experiment_metadata": case.get("experiment_metadata", {}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-judge eval for pathway env")
    p.add_argument("--case", required=True, help="Case file path relative to data dir")
    p.add_argument(
        "--reference-dir",
        required=False,
        default=None,
        help="Deprecated: reference is now built from live episode outputs",
    )
    p.add_argument("--agent-model", default="gpt-4o-mini")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--out-json", default=None, help="Optional path to save artifacts")
    args = p.parse_args()
    _load_dotenv()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
