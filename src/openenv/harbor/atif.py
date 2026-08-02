"""Reconcile our token capture against Harbor's ATIF trajectory. The independent cross-check.

Harbor agents write `agent/trajectory.json` in ATIF (Agent Trajectory Interchange Format, currently
v1.7), a published spec for logging agent interaction histories across debugging, SFT and RL. ~27 of
Harbor's agents build ATIF trajectories, which is far more than the six the docs list.

Why this matters more than it sounds. ATIF and the intercept measure the same rollout through
completely independent paths: the harness counts its own tokens and reports them to Harbor, while we
derive ours from engine-returned token ids stitched along a graph. If they agree turn by turn, the
masking, the turn segmentation and the prefix stitching are all correct simultaneously. Nothing else
we can run gives that assurance, because every internal check shares our own assumptions.

Validated on opencode + Qwen3.5-4B, 8 agent steps:

    ATIF completion_tokens : [37, 36, 104, 264, 255, 119, 32, 27]   total 874
    intercept turn_lengths : [37, 36, 104, 264, 255, 119, 32, 27]   total 874
    ATIF step-1 prompt_tokens 7990  ==  intercept prompt_len 7990

ATIF also carries three things a proxy structurally cannot see, which is the other half of the value:

    llm_call_count          >1 means the harness burned several model calls on one logical step,
                            i.e. it retried. A proxy sees the calls but not that they were retries.
    subagent_trajectories   nested trajectories (v1.7). Ground truth for which turns are a subagent,
                            instead of inferring it from graph roots.
    tool_call_id <-> observation.source_call_id
                            which tool result answered which call.

`Metrics` in ATIF has optional `logprobs` and `completion_token_ids` fields, which harnesses leave
empty. So the end state is not two formats to reconcile: it is ATIF with our token fields filled in,
one artifact that is trace, SFT dataset and RL data at once. `merge_into_atif` does that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openenv.core.harness.capture.validate import FATAL, INFO, Report, WARN


def load_atif(trial_dir: str | Path) -> dict[str, Any] | None:
    """Read `agent/trajectory.json` from a Harbor trial dir. None if the agent emitted none."""
    path = Path(trial_dir) / "agent" / "trajectory.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - a malformed trace must not break a good rollout
        return None


def agent_steps(atif: dict[str, Any]) -> list[dict[str, Any]]:
    """Steps the agent produced. `source` is one of user | agent | system."""
    return [s for s in (atif.get("steps") or []) if s.get("source") == "agent"]


def atif_turn_lengths(atif: dict[str, Any]) -> list[int]:
    return [
        int((s.get("metrics") or {}).get("completion_tokens") or 0)
        for s in agent_steps(atif)
    ]


def _subsequence_gap(needle: list[int], haystack: list[int]) -> list[int] | None:
    """Indices of `haystack` skipped when `needle` is matched as a subsequence, else None.

    Greedy two-pointer, which is exact for subsequence membership. Returns None the moment a needle
    element cannot be found, so a genuine disagreement (a value we never captured, or counts that
    differ) falls through to the FATAL path rather than being explained away as auxiliary calls.

    Requires the needle to be strictly shorter; equal lists are handled by the exact-match path and
    a LONGER needle means ATIF logged calls we never saw, which is a capture failure, not aux calls.
    """
    if len(needle) >= len(haystack):
        return None
    skipped: list[int] = []
    j = 0
    for i, value in enumerate(haystack):
        if j < len(needle) and value == needle[j]:
            j += 1
        else:
            skipped.append(i)
    return skipped if j == len(needle) else None


def reconcile(document: dict[str, Any], atif: dict[str, Any] | None) -> Report:
    """Compare the exported rollout against ATIF. Disagreement is the signal.

    Deliberately FATAL on a per-turn mismatch. Two independent measurements of the same rollout
    disagreeing means one is wrong, and we cannot tell which. Training on data we cannot corroborate
    is exactly the failure this whole layer exists to prevent.
    """
    report = Report()
    if atif is None:
        report.add(
            INFO, "no_atif", "agent emitted no ATIF trajectory; cross-check unavailable"
        )
        return report

    report.add(
        INFO,
        "atif_version",
        f"schema {atif.get('schema_version')} "
        f"agent {(atif.get('agent') or {}).get('name')}",
    )

    agent_rows = [r for r in document["sequences"] if r["role"] == "agent"]
    if not agent_rows:
        # When the intercept saw NO calls at all, `check_rollout` already says so plainly. Adding a
        # second FATAL here just buries the real cause: I misread this as a distinct failure mode
        # twice tonight before noticing it was always downstream of `no_turns`.
        if not document.get("turns"):
            report.add(
                INFO,
                "no_turns_upstream",
                "no model calls were captured, so there is nothing to reconcile; see the "
                "rollout's own no_turns finding for the cause",
            )
        else:
            report.add(
                FATAL,
                "no_agent_sequence",
                f"{len(document['turns'])} calls captured but none labelled 'agent'",
            )
        return report

    # Compare against EVERY call the agent made, in arrival order, across ALL agent roots.
    #
    # Two reasons this is the right comparison rather than the surviving training path:
    #   * ATIF logs every LLM call including retries, so a rollout that retried would report a false
    #     mismatch. Comparing the full list also validates the discard decisions: get one wrong and
    #     the equality breaks.
    #   * A harness that rewrites its system prompt mid-run (claude-code) splits one conversation
    #     across several roots. ATIF still sees one flat list, so the union of agent roots is what
    #     lines up with it.
    agent_roots = {r["root_id"] for r in agent_rows}
    turns = [t for t in document.get("turns", []) if t["root_id"] in agent_roots]
    ours_all = [t["n_sampled"] for t in turns]
    theirs = atif_turn_lengths(atif)

    # A converter that never fills in token counts gives us nothing to compare against. vibe reports
    # completion_tokens=0 on every step while capturing perfectly (reward 1.0, 6 turns, 1197
    # trainable tokens), so calling that a MISMATCH would fail a harness for its trace converter's
    # laziness rather than for anything wrong with the rollout. Downgrade to "no cross-check
    # available", which is what `atif=none` already means for harnesses that emit nothing at all.
    if not any(theirs):
        # Covers BOTH shapes of an unhelpful converter: all-zero counts (vibe) and no agent steps at
        # all (antigravity-sdk reported `[]` while we captured a 145-token call). Either way there is
        # nothing to compare against, and failing the rollout would punish it for the trace
        # converter's gaps rather than for anything wrong with the capture.
        detail = (
            f"{len(theirs)} steps but all completion_tokens are 0"
            if theirs
            else "no agent steps at all"
        )
        report.add(
            WARN,
            "atif_no_token_counts",
            f"ATIF has {detail}, so no independent cross-check is possible. Intercept "
            f"captured {sum(ours_all)} tokens across {len(ours_all)} calls; those numbers "
            "stand unverified.",
        )
        return report

    # ATIF being a strict SUBSEQUENCE of our calls is a real and benign shape, distinct from a
    # disagreement. It means we saw calls the harness did not log as agent steps, which is what an
    # auxiliary call is: gemini-cli fires a "next speaker" check between steps, mini-swe-agent does
    # something similar. Both measurements are individually right.
    #
    #   intercept: [58, 132, 266, 370, 105, 32, 54, 164, 33]
    #   ATIF:      [58, 132, 266, 370, 105, 32,          33]
    #
    # The old code called any inequality FATAL, which failed a rollout for correctly capturing MORE
    # than the harness chose to log. Note the asymmetry that makes this safe: extra calls on OUR side
    # are explainable, whereas MISSING calls would mean we lost something, and that still fails.
    #
    # The aux calls are identified by position and excluded from training, because they are not the
    # agent working on the task and must not carry the rollout's reward.
    aux_idx = _subsequence_gap(theirs, ours_all) if ours_all != theirs else None

    # A subsequence match only carries evidence when ATIF accounts for MOST of what we captured.
    # The shorter the ATIF list relative to ours, the more likely a match is coincidence: 5 values
    # will embed in 49 almost by construction, so "the other 44 are auxiliary" is an inference the
    # data does not support.
    #
    # Real example that forced this. mimo on one task: 49 captured calls, ATIF logged 5, and the
    # matcher happily demoted 44 to auxiliary under a WARN, discarding 90% of a rollout without
    # failing anything. Contrast the cases where the inference IS sound, where ATIF covers the large
    # majority: gemini-cli 8/12, 14/19, 6/10, mini-swe-agent 5/6.
    #
    # Below the floor we refuse rather than guess. Silently training on a tenth of a rollout is a
    # worse outcome than an explicit failure, which is the whole premise of this layer.
    if aux_idx is not None and len(theirs) < 0.5 * len(ours_all):
        report.add(
            FATAL,
            "atif_coverage_too_low",
            f"ATIF accounts for only {len(theirs)} of {len(ours_all)} captured calls "
            f"({len(theirs) / len(ours_all):.0%}). They embed as a subsequence, but at this "
            "ratio that is as likely coincidence as signal, so the extra calls cannot be "
            "called auxiliary with any confidence. Refusing rather than discarding "
            f"{len(aux_idx)} calls on a guess.\n"
            f"        intercept: {ours_all}\n        ATIF     : {theirs}",
        )
        return report

    if aux_idx is not None:
        aux_nodes = [turns[i]["node_id"] for i in aux_idx]
        report.add(
            WARN,
            "atif_aux_calls",
            f"{len(aux_idx)} of {len(ours_all)} captured calls are absent from ATIF, so the "
            f"harness did not consider them agent steps (auxiliary calls such as "
            f"gemini-cli's next-speaker check). The remaining {len(theirs)} agree "
            f"token-for-token. Sizes: {[ours_all[i] for i in aux_idx]}. These are excluded "
            f"from training rather than credited with the rollout's reward.",
        )
        report.aux_node_ids = aux_nodes
        return report

    if ours_all == theirs:
        n_discarded = sum(1 for t in turns if t["discarded"])
        n_trained = sum(len(r["turn_lengths"]) for r in agent_rows)
        detail = (
            f"all {len(ours_all)} calls agree token-for-token (total {sum(ours_all)}); "
            f"{n_discarded} discarded, {n_trained} trained"
        )
        if len(agent_rows) > 1:
            detail += (
                f"; across {len(agent_rows)} agent sequences (the harness rewrote its prompt "
                "mid-run, so the conversation spans several token-prefix families)"
            )
        report.add(INFO, "turns_match", detail)
    else:
        report.add(
            FATAL,
            "turn_mismatch",
            f"per-call completion tokens disagree.\n"
            f"        intercept (all calls): {ours_all}\n"
            f"        ATIF                 : {theirs}\n"
            f"        intercept (trained)  : {[r['turn_lengths'] for r in agent_rows]}",
        )

    steps = agent_steps(atif)
    if steps:
        atif_prompt = int((steps[0].get("metrics") or {}).get("prompt_tokens") or 0)
        # The first agent sequence in arrival order holds the rollout's opening prompt.
        first_prompt_len = agent_rows[0]["prompt_len"]
        if atif_prompt and atif_prompt != first_prompt_len:
            report.add(
                WARN,
                "prompt_len_mismatch",
                f"first-turn prompt: intercept {first_prompt_len} vs ATIF {atif_prompt}",
            )

    retried = [s for s in steps if int(s.get("llm_call_count") or 1) > 1]
    if retried:
        report.add(
            WARN,
            "atif_retries",
            f"{len(retried)} ATIF step(s) report llm_call_count>1: the harness retried. "
            f"Graph found {document['stats']['n_discarded']} discarded turn(s).",
        )

    subagents = atif.get("subagent_trajectories") or []
    if subagents:
        report.add(
            WARN,
            "atif_subagents",
            f"ATIF reports {len(subagents)} subagent trajectory(ies); graph has "
            f"{document['stats']['n_roots']} roots. Subagent turns must not be trained with "
            "the parent rollout's reward.",
        )
    return report


def merge_into_atif(atif: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """Fill ATIF's empty `completion_token_ids` / `logprobs` with our captured values.

    Produces one artifact that is both the human-readable trace and the training data, rather than
    two formats a consumer has to join. Only attempted when the per-turn counts already agree; a
    mismatch means we cannot map our tokens onto their steps, and guessing would be worse than
    leaving the fields empty.
    """
    agent_rows = [r for r in document["sequences"] if r["role"] == "agent"]
    if not agent_rows:
        return atif

    # ATIF logs every call the harness made; our training rows contain only the ones that survived,
    # possibly split across several roots. Align on the FULL call list in arrival order and skip the
    # discarded positions, rather than zipping lists of different length and silently shifting every
    # token onto the wrong step.
    agent_roots = {r["root_id"] for r in agent_rows}
    turns_meta = [t for t in document.get("turns", []) if t["root_id"] in agent_roots]
    steps_all = [s for s in atif.get("steps") or [] if s.get("source") == "agent"]
    if len(turns_meta) != len(steps_all):
        return atif
    if [t["n_sampled"] for t in turns_meta] != atif_turn_lengths(atif):
        return atif

    merged = json.loads(json.dumps(atif))  # never mutate Harbor's artifact in place
    steps = [s for s in merged["steps"] if s.get("source") == "agent"]

    # Sampled tokens from every agent sequence, in the order their nodes arrived.
    by_node: dict[str, list[tuple[int, float]]] = {}
    for row in agent_rows:
        sampled = [
            (tid, lp)
            for tid, m, lp in zip(row["input_ids"], row["loss_mask"], row["logprobs"])
            if m
        ]
        # One mask-run per node that contributed trainable tokens. If a node was masked out (its
        # logprobs could not be trusted) the counts diverge and the node->span mapping is no longer
        # reliable, so we decline to merge rather than attach tokens to the wrong step.
        if len(row["node_ids"]) != len(row["turn_lengths"]):
            return atif
        cursor = 0
        for node_id, length in zip(row["node_ids"], row["turn_lengths"]):
            by_node[node_id] = sampled[cursor : cursor + length]
            cursor += length

    for meta, step in zip(turns_meta, steps):
        metrics = step.setdefault("metrics", {})
        if meta["discarded"]:
            # Generated, then abandoned by the harness. Recorded so the trace stays complete, but
            # flagged so nobody trains it with the rollout's reward.
            metrics["discarded"] = True
            continue
        span = by_node.get(meta["node_id"])
        if span is None:
            continue
        metrics["completion_token_ids"] = [t for t, _ in span]
        metrics["logprobs"] = [lp for _, lp in span]

    first = agent_rows[0]
    merged.setdefault("extra", {})["intercept"] = {
        "prompt_ids": first["input_ids"][: first["prompt_len"]],
        "n_trainable": sum(r["n_trainable"] for r in agent_rows),
        "n_agent_sequences": len(agent_rows),
        "session_id": document["session_id"],
    }
    return merged
