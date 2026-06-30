#!/usr/bin/env python3
"""
Tool-calling LLM agent evaluation for pathway_analysis_env.

Runs a tool-calling LLM agent over ``data/eval_manifest.json``. The agent is
given the pathway tools and decides which to call and when to submit_answer.
Writes JSON + Markdown reports.

Free providers (no credit card):
  Groq:       export GROQ_API_KEY=...  (https://console.groq.com)
  OpenRouter: export OPENROUTER_API_KEY=...  (model openrouter/free)
  Ollama:     ollama serve && ollama pull llama3.1:8b  (--provider ollama)

Usage:
  export GROQ_API_KEY=...
  export MPLCONFIGDIR=/tmp/mpl
  PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/run_llm_agent_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pathway_analysis_env.agent_openai_tools import (
    OPENAI_TOOLS,
    observation_to_tool_result_content,
    tool_call_to_pathway_action,
)
from pathway_analysis_env.server.analysis import gseapy_available, pydeseq2_available
from pathway_analysis_env.server.pathway_environment import DATA_DIR, PathwayEnvironment

DEFAULT_SYSTEM_PROMPT = """You are a computational biologist agent operating a pathway analysis environment.

Required workflow (eval mode):
1. understand_experiment_design and/or inspect_dataset — learn groups and sample layout.
2. run_differential_expression — set reference (baseline) vs alternate (treatment) conditions.
3. run_pathway_enrichment — ORA on DE genes (do not pass a custom gene_list).
4. Optionally compare_pathways between two top pathway names.
5. submit_answer — one pathway hypothesis string supported by ORA.

Rules:
- Never guess without running DE and ORA first.
- Use condition names exactly as returned in available_conditions.
- For submit_answer, name a specific pathway (e.g. from top_pathways), not a long essay.
"""

MINIMAL_SYSTEM_PROMPT = """Run pathway workflow: design/inspect → DE (reference vs alternate) → ORA → submit_answer with one pathway from ORA. Use exact condition names."""

# OpenAI-compatible providers (Groq is free — no credit card).
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_models": ["llama-3.3-70b-versatile"],
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_models": ["openrouter/free"],
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "default_models": ["gpt-4o-mini", "gpt-4o"],
    },
    "ollama": {
        "api_key_env": None,
        "base_url": "http://127.0.0.1:11434/v1",
        "default_models": ["llama3.1:8b"],
    },
}


@dataclass
class LLMProvider:
    name: str
    api_key: str
    base_url: Optional[str]
    default_models: List[str]


@dataclass
class EpisodeResult:
    agent_id: str
    model: str
    episode_id: str
    case_file: str
    passed: bool
    score: float
    steps: int
    turns: int
    wall_time_s: float
    done: bool
    hypothesis: Optional[str] = None
    match_mode: Optional[str] = None
    failure_code: Optional[str] = None
    action_trace: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "episode_id": self.episode_id,
            "case_file": self.case_file,
            "passed": self.passed,
            "score": self.score,
            "steps": self.steps,
            "turns": self.turns,
            "wall_time_s": round(self.wall_time_s, 2),
            "done": self.done,
            "hypothesis": self.hypothesis,
            "match_mode": self.match_mode,
            "failure_code": self.failure_code,
            "action_trace": self.action_trace,
            "error": self.error,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


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


def _ollama_reachable() -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=1.5
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def resolve_llm_provider(explicit: str = "auto") -> Optional[LLMProvider]:
    """Pick an LLM backend from env vars or an explicit --provider flag."""
    order = ["groq", "openrouter", "openai", "ollama"]
    names = [explicit] if explicit != "auto" else order

    for name in names:
        if name not in LLM_PROVIDERS:
            raise SystemExit(
                f"Unknown provider {name!r}. Choose: auto, {', '.join(order)}"
            )
        spec = LLM_PROVIDERS[name]
        key_env = spec.get("api_key_env")
        if key_env:
            api_key = os.environ.get(key_env, "")
            if not api_key:
                continue
        elif name == "ollama":
            if not _ollama_reachable():
                continue
            api_key = "ollama"
        else:
            continue
        return LLMProvider(
            name=name,
            api_key=api_key,
            base_url=spec.get("base_url"),
            default_models=list(spec["default_models"]),
        )
    return None


def make_llm_client(provider: LLMProvider):
    from openai import AsyncOpenAI

    kwargs: Dict[str, Any] = {"api_key": provider.api_key}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    return AsyncOpenAI(**kwargs)


def _episode_skipped(spec: Dict[str, Any]) -> Optional[str]:
    if spec.get("requires_pydeseq2") and not pydeseq2_available():
        return "pydeseq2_unavailable"
    if spec.get("requires_gseapy") and not gseapy_available():
        return "gseapy_unavailable"
    return None


def _skipped_result(
    agent_id: str, model: str, spec: Dict[str, Any], reason: str
) -> EpisodeResult:
    return EpisodeResult(
        agent_id=agent_id,
        model=model,
        episode_id=spec["id"],
        case_file=spec["case_file"],
        passed=False,
        score=0.0,
        steps=0,
        turns=0,
        wall_time_s=0.0,
        done=False,
        skipped=True,
        skip_reason=reason,
    )


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """Parse a provider 'try again in Xs' hint from a rate-limit error."""
    import re

    text = str(exc)
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)s", text)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2)) if m.group(2) else 0.0
    return minutes * 60.0 + seconds


def _supports_temperature_override(model: str) -> bool:
    """Some models (e.g. gpt-5) only support default temperature."""
    return not model.startswith("gpt-5")


async def _chat_with_retry(
    client,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int,
):
    """Call chat.completions with backoff on rate limits / transient errors.

    Honors the provider's "try again in Xs" hint when present; otherwise uses
    exponential backoff. Tool-use parser hiccups (Groq ``tool_use_failed``) are
    also retried since they are non-deterministic.
    """
    attempt = 0
    tool_hiccups = 0
    while True:
        # Base call is deterministic (T=0). On a provider tool-call parser
        # failure, nudge temperature up so retries are not identical (and thus
        # not guaranteed to fail the same way).
        temperature = min(0.2 * tool_hiccups, 0.8)
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": OPENAI_TOOLS,
                "tool_choice": "auto",
            }
            if _supports_temperature_override(model):
                kwargs["temperature"] = temperature
            return await client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider-agnostic retry
            text = str(exc)
            is_rate_limit = "429" in text or "rate_limit" in text.lower()
            is_tool_hiccup = "tool_use_failed" in text
            if attempt >= max_retries or not (is_rate_limit or is_tool_hiccup):
                raise
            if is_tool_hiccup:
                tool_hiccups += 1
            hinted = _retry_after_seconds(exc) if is_rate_limit else None
            if hinted is not None:
                delay = hinted + 1.0  # cushion past the rate-limit window
            elif is_tool_hiccup:
                delay = 1.0  # parser hiccup: retry quickly with new temperature
            else:
                delay = min(2.0 * (2**attempt), 60.0)
            print(
                f"    retry {attempt + 1}/{max_retries} after "
                f"{'rate limit' if is_rate_limit else 'tool_use_failed'} "
                f"(sleeping {delay:.1f}s, temp->{min(0.2 * tool_hiccups, 0.8):.1f})...",
                flush=True,
            )
            await asyncio.sleep(delay)
            attempt += 1


async def run_llm_episode(
    *,
    agent_id: str,
    model: str,
    case_file: str,
    episode_id: str,
    system_prompt: str,
    max_turns: int,
    strict: bool,
    provider: LLMProvider,
    max_retries: int = 6,
) -> EpisodeResult:
    t0 = time.perf_counter()
    action_trace: List[str] = []
    client = make_llm_client(provider)
    env = PathwayEnvironment(case_file=case_file)
    try:
        obs = env.reset(orchestrator_mode=True, strict=strict)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Episode {episode_id}. Case: {case_file}. "
                    f"Conditions: {obs.available_conditions}. {obs.message}"
                ),
            },
        ]
        last_failure: Optional[str] = None
        turn = 0
        for turn in range(max_turns):
            response = await _chat_with_retry(
                client,
                model=model,
                messages=messages,
                max_retries=max_retries,
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                if env.state.is_done:
                    break
                continue
            # Reconstruct a clean assistant message with only fields that
            # OpenAI-compatible providers universally accept. The OpenAI SDK
            # adds extra fields (e.g. ``annotations``) that strict providers
            # such as Groq reject with a 400 error on the next request.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                action = tool_call_to_pathway_action(
                    name=tc.function.name,
                    arguments_json=tc.function.arguments,
                )
                action_trace.append(action.action_type)
                step_obs = env.step(action)
                meta = step_obs.metadata or {}
                if meta.get("failure_code"):
                    last_failure = str(meta["failure_code"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation_to_tool_result_content(step_obs),
                    }
                )
                if step_obs.done:
                    break
            if env.state.is_done:
                break
        outcome = env.episode_outcome or {}
        return EpisodeResult(
            agent_id=agent_id,
            model=model,
            episode_id=episode_id,
            case_file=case_file,
            passed=bool(outcome.get("correct")),
            score=float(outcome.get("score") or 0.0),
            steps=env.state.step_count,
            turns=turn + 1,
            wall_time_s=time.perf_counter() - t0,
            done=env.state.is_done,
            hypothesis=outcome.get("hypothesis"),
            match_mode=outcome.get("match_mode"),
            failure_code=last_failure if not outcome.get("correct") else None,
            action_trace=action_trace,
        )
    except Exception as exc:
        return EpisodeResult(
            agent_id=agent_id,
            model=model,
            episode_id=episode_id,
            case_file=case_file,
            passed=False,
            score=0.0,
            steps=0,
            turns=0,
            wall_time_s=time.perf_counter() - t0,
            done=False,
            error=f"{type(exc).__name__}: {exc}",
            action_trace=action_trace,
        )


def aggregate(results: List[EpisodeResult]) -> Dict[str, Any]:
    by_agent: Dict[str, List[EpisodeResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent_id, []).append(r)
    agents_summary = []
    for agent_id, rows in sorted(by_agent.items()):
        run_rows = [x for x in rows if not x.skipped]
        passed = sum(1 for x in run_rows if x.passed)
        agents_summary.append(
            {
                "agent_id": agent_id,
                "model": rows[0].model if rows else "",
                "episodes_run": len(run_rows),
                "episodes_passed": passed,
                "pass_rate": passed / len(run_rows) if run_rows else 0.0,
                "avg_score": (
                    sum(x.score for x in run_rows) / len(run_rows) if run_rows else 0.0
                ),
                "avg_steps": (
                    sum(x.steps for x in run_rows) / len(run_rows) if run_rows else 0.0
                ),
                "avg_wall_time_s": (
                    sum(x.wall_time_s for x in run_rows) / len(run_rows)
                    if run_rows
                    else 0.0
                ),
            }
        )
    return {"agents": agents_summary}


def write_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Pathway Agent Evaluation Report",
        "",
        f"Generated: {summary.get('generated_at', '')}",
        "",
        "## Eval plan",
        "",
        summary.get("eval_plan", ""),
        "",
        "## Leaderboard",
        "",
        "| Agent | Model | Pass rate | Avg score | Avg steps | Avg time (s) |",
        "|-------|-------|-----------|-----------|-----------|--------------|",
    ]
    for a in summary.get("aggregate", {}).get("agents", []):
        lines.append(
            f"| {a['agent_id']} | {a['model']} | {a['pass_rate']:.0%} "
            f"({a['episodes_passed']}/{a['episodes_run']}) | {a['avg_score']:.2f} | "
            f"{a['avg_steps']:.1f} | {a['avg_wall_time_s']:.1f} |"
        )
    lines.extend(["", "## Per-episode results", ""])
    for r in summary.get("results", []):
        status = "SKIP" if r.get("skipped") else ("PASS" if r.get("passed") else "FAIL")
        lines.append(
            f"- **{status}** `{r.get('agent_id')}` / `{r.get('episode_id')}` "
            f"— score={r.get('score')} steps={r.get('steps')} "
            f"hypothesis={r.get('hypothesis')!r} "
            f"failure={r.get('failure_code') or r.get('error') or '—'}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[LLMProvider]]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    provider = resolve_llm_provider(args.provider)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models and provider:
        models = list(provider.default_models)

    eval_plan = (
        "Tool-calling LLM agent at T=0 over the manifest episodes "
        "(provider: Groq/OpenRouter/OpenAI/Ollama). The agent is given the "
        "pathway tools and decides which to call and when to submit_answer. "
        "Metrics: pass rate, avg score, steps, wall time, action trace, failure codes."
    )

    results: List[EpisodeResult] = []

    if provider is None:
        print(
            "No LLM API key found — cannot run the agent.\n"
            "  Free option: export GROQ_API_KEY=...  (sign up at https://console.groq.com)\n"
            "  Or: ollama serve && --provider ollama\n"
            "  Or add GROQ_API_KEY to repo-root .env",
            flush=True,
        )
    else:
        print(f"LLM provider: {provider.name}  models: {', '.join(models)}", flush=True)
        prompt_variants = [("llm_default", DEFAULT_SYSTEM_PROMPT)]
        if args.prompt_ablation:
            prompt_variants.append(("llm_minimal", MINIMAL_SYSTEM_PROMPT))
        for model in models:
            for prompt_name, prompt_text in prompt_variants:
                agent_id = f"{prompt_name}__{model.replace('/', '_').replace(':', '_')}"
                for spec in episodes:
                    skip = _episode_skipped(spec)
                    if skip:
                        results.append(_skipped_result(agent_id, model, spec, skip))
                        continue
                    print(f"Running {agent_id} on {spec['id']}...", flush=True)
                    r = await run_llm_episode(
                        agent_id=agent_id,
                        model=model,
                        case_file=spec["case_file"],
                        episode_id=spec["id"],
                        system_prompt=prompt_text,
                        max_turns=args.max_turns,
                        strict=args.strict,
                        provider=provider,
                        max_retries=args.max_retries,
                    )
                    results.append(r)
                    print(
                        f"  -> {'PASS' if r.passed else 'FAIL'} score={r.score} steps={r.steps}",
                        flush=True,
                    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_plan": eval_plan,
        "manifest": str(args.manifest),
        "llm_provider": provider.name if provider else None,
        "models": models if provider else [],
        "aggregate": aggregate(results),
        "results": [r.to_dict() for r in results],
    }
    return summary, provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tool-calling LLM agent eval for pathway env"
    )
    parser.add_argument(
        "--manifest", type=Path, default=DATA_DIR / "eval_manifest.json"
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "groq", "openrouter", "openai", "ollama"],
        help="LLM backend (auto tries Groq, OpenRouter, OpenAI, then Ollama)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs (defaults per provider if omitted)",
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retries per LLM call on rate-limit / transient tool errors",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--prompt-ablation", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("envs/pathway_analysis_env/outputs/llm_eval/latest.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("envs/pathway_analysis_env/outputs/llm_eval/latest.md"),
    )
    args = parser.parse_args()

    _load_dotenv()

    try:
        summary, _provider = asyncio.run(main_async(args))
    except Exception:
        traceback.print_exc()
        raise

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(summary, args.md_out)
    print(f"\nWrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    print("\nLeaderboard:")
    for a in summary["aggregate"]["agents"]:
        print(
            f"  {a['agent_id']:40s} pass={a['pass_rate']:.0%} "
            f"avg_score={a['avg_score']:.2f} avg_steps={a['avg_steps']:.1f}"
        )


if __name__ == "__main__":
    main()
