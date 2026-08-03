"""Run one Harbor trial through the capture proxy and return trainable output.

This is where every piece meets: a task from the dataset, a harness from the seam table, a sandbox
from Harbor, and a capture session whose id doubles as the agent's API key.

    task dir ──┐
    harness ───┼──> TrialConfig ──> Trial.run() ──> TrialResult (reward)
    sandbox ───┘                         │
                                         └──> agent talks to the intercept
                                              (api key == session id)
                                                     │
                                              RolloutGraph ──> HarborRolloutResult

**Nothing raises out of `run_rollout`.** A failed rollout returns `ok=False` with `reward=None`. That
is not defensive habit, it is the reason this layer exists: in the white-box predecessor a rollout
exception reached the trainer and hung every rank at the NCCL barrier forever, so every method there
had to be individually wrapped. Behind a result object that failure mode cannot occur.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from openenv.core.harness.capture.export import export_session

from . import seams
from .atif import load_trace, reconcile
from .models import (
    conversations_from_document,
    HarborRolloutResult,
    HarborStepResult,
    HarborTurn,
    turns_from_document,
)

# Harbor's own retry is deliberately off. A Harbor-level retry re-runs the agent against the SAME
# capture session, so two attempts merge into one graph and the trace cross-check compares a
# two-attempt capture against a one-attempt trajectory. One attempt = one session = one rollout;
# retries belong to the caller, with a fresh session each time.
_NO_HARBOR_RETRY = 0


# Some agents read os.environ at CONSTRUCTION, before any sandbox exists — Harbor's claude-code
# wrapper decides there which credentials to forward. For those, `agent_env` alone is too late, so
# the seam also carries a process-env channel.
#
# That channel is global, so two concurrent rollouts of the same harness would overwrite each
# other's session key. The lock is held only across agent construction (`Trial.create`), which is
# brief; the rollout itself then runs unserialised.
#
# A `threading.Lock`, not an `asyncio.Lock`. What is being protected is `os.environ`, which is global
# to the PROCESS, not to an event loop, and rollouts arrive on several loops: the env server answers
# each request on its own loop, and any caller using `asyncio.run` per rollout creates another. An
# asyncio lock binds to the first loop that uses it and then raises
# "is bound to a different event loop" for every other one, which fails 100% of concurrent rollouts
# while passing every sequential test.
_PROC_ENV_LOCK = threading.Lock()


def apply_process_env(seam_name: str, env: dict[str, str]) -> None:
    """Set the seam's process-level env vars, warning when one would break the grader.

    OPENAI_API_KEY is shared with the task grader: Harbor forwards it into the sandbox and the
    DataAgent grader's LLM-judge tier fires on `if os.environ.get("OPENAI_API_KEY")`. Overwriting it
    with a session id makes the judge 401, so every semantically-correct-but-not-exact answer scores
    0 — which reads as a weak model rather than a broken harness, and poisons the RL baseline.
    """
    grader_key = os.environ.get("OPENAI_API_KEY")
    for key, value in env.items():
        if key == "OPENAI_API_KEY" and grader_key and value != grader_key:
            print(
                f"[{seam_name}] WARNING: this seam overwrites OPENAI_API_KEY, which the grader "
                "uses for its LLM-judge tier. Judging is disabled for this run (exact-match and "
                "numeric tolerance still apply)."
            )
        os.environ[key] = value


def _pick_reward(
    rewards: dict[str, float], reward_key: str = ""
) -> tuple[float | None, str]:
    """Collapse Harbor's reward dict to the one scalar an RL trainer consumes.

    Refuses rather than guesses. Harbor lets a task emit any keys it likes, and inventing a rule to
    combine them is inventing reward semantics — which is exactly how a previous run got reward
    hacked: a `+0.2 for submitting anything` term made the policy learn to quick-submit, training
    reward looked healthy, and eval collapsed from 0.740 to 0.178.

    Args:
        rewards (`dict[str, float]`):
            The verifier's reward dict, as Harbor produced it.
        reward_key (`str`, *optional*):
            Force a specific key. Required when the dict has several and none is named `reward`.

    Returns:
        `tuple[float | None, str]`: The chosen value and the key it came from.

    Raises:
        ValueError: If the key is ambiguous and none was given.
    """
    if not rewards:
        return None, ""
    if reward_key:
        if reward_key not in rewards:
            raise ValueError(f"reward_key {reward_key!r} not in {sorted(rewards)}")
        return float(rewards[reward_key]), reward_key
    if len(rewards) == 1:
        key = next(iter(rewards))
        return float(rewards[key]), key
    if "reward" in rewards:
        return float(rewards["reward"]), "reward"
    raise ValueError(
        f"task produced several rewards {sorted(rewards)} and none is named 'reward'. "
        "Pass reward_key= to say which one is the training signal."
    )


def build_trial_config(
    *,
    task_dir: Path | str,
    harness: str,
    sandbox: str,
    intercept_url: str,
    session_id: str,
    model: str,
    trial_name: str,
    trials_dir: Path | str,
    keep_sandbox: bool = False,
    agent_timeout_sec: float | None = None,
    force_build: bool = False,
) -> Any:
    """Assemble Harbor's `TrialConfig` for one rollout.

    The seam decides how this particular agent learns about the proxy — an env var, a constructor
    kwarg, or a config file written by a subclass. That is the only per-agent knowledge involved;
    everything downstream is identical.
    """
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
        VerifierConfig,
    )

    seam = seams.get(harness)
    model_name, kwargs, agent_env, proc_env = seam.resolve(
        base_url=intercept_url, session=session_id, model=model
    )
    apply_process_env(seam.name, proc_env)

    agent = AgentConfig(
        name=seam.import_path or harness,
        import_path=seam.import_path,
        model_name=model_name,
        kwargs=kwargs,
        env=agent_env,
        override_timeout_sec=agent_timeout_sec,
    )
    return TrialConfig(
        task=TaskConfig(path=Path(str(task_dir))),
        agent=agent,
        environment=EnvironmentConfig(
            type=sandbox, delete=not keep_sandbox, force_build=force_build
        ),
        verifier=VerifierConfig(),
        trial_name=trial_name,
        trials_dir=Path(str(trials_dir)),
    )


async def run_rollout(
    *,
    task_dir: Path | str,
    harness: str,
    sandbox: str,
    registry: Any,
    intercept_url: str,
    model: str,
    trials_dir: Path | str,
    dataset: str = "",
    reward_key: str = "",
    keep_sandbox: bool = False,
    agent_timeout_sec: float | None = None,
    force_build: bool = False,
    session_prefix: str = "oe",
) -> HarborRolloutResult:
    """Run one rollout end to end. Never raises.

    Args:
        task_dir (`Path` or `str`):
            The Harbor task directory to run.
        harness (`str`):
            A seam name (`opencode`, `claude-code`, ...) or a `module:Class` import path.
        sandbox (`str`):
            A Harbor `EnvironmentType`, e.g. `e2b` or `modal`.
        registry (`SessionRegistry`):
            The live capture registry; a session is minted here and its id becomes the agent's key.
        intercept_url (`str`):
            Public URL of the capture proxy, as the sandbox must reach it.
        model (`str`):
            Served model id. Normalised per harness by the seam.
        trials_dir (`Path` or `str`):
            Where Harbor writes trial artifacts.
        reward_key (`str`, *optional*):
            Which reward key is the training signal, for multi-reward tasks.
        keep_sandbox (`bool`, *optional*, defaults to `False`):
            Leave the sandbox alive after the run, for debugging.

    Returns:
        [`HarborRolloutResult`]: Reward, per-turn token ids and logprobs, and validation findings.
    """
    task_dir = Path(str(task_dir))
    started = time.monotonic()

    # The session id lands in Harbor's E2B sandbox metadata (it is derived from trial_name), which
    # is what later makes it possible to reap only the sandboxes this server created.
    trial_name = f"{session_prefix}-{task_dir.name[:28]}-{uuid.uuid4().hex[:8]}"
    session = registry.create(
        session_id=None, harness=harness, sandbox=sandbox, task=task_dir.name
    )

    result = HarborRolloutResult(
        task_id=str(task_dir),
        task_name=task_dir.name,
        dataset=dataset,
        harness=harness,
        sandbox=sandbox,
        trial_name=trial_name,
        session_id=session.session_id,
    )

    trial_result = None
    try:
        from harbor.trial.trial import Trial

        config_kwargs = dict(
            task_dir=task_dir,
            harness=harness,
            sandbox=sandbox,
            intercept_url=intercept_url,
            session_id=session.session_id,
            model=model,
            trial_name=trial_name,
            trials_dir=trials_dir,
            keep_sandbox=keep_sandbox,
            agent_timeout_sec=agent_timeout_sec,
            force_build=force_build,
        )

        # Held across construction because Harbor's wrappers read `os.environ` while building the
        # agent, so the variables must still be in place when `Trial.create` runs. It is a blocking
        # acquire inside an async function, which is acceptable only because construction does no
        # I/O worth speaking of: the sandbox is booted later, by `trial.run()`, outside the lock.
        with _PROC_ENV_LOCK:
            config = build_trial_config(**config_kwargs)
            trial = await Trial.create(config)
        trial_result = await trial.run()
    except Exception as exc:  # noqa: BLE001 - a rollout failure is a RESULT, never an exception
        result.ok = False
        result.error = str(exc)[:600]
        result.exception_type = type(exc).__name__

    result.wall_s = round(time.monotonic() - started, 2)

    # --- reward, forwarded verbatim -------------------------------------
    if trial_result is not None:
        verifier = getattr(trial_result, "verifier_result", None)
        rewards = dict(getattr(verifier, "rewards", None) or {}) if verifier else {}
        result.rewards = {k: float(v) for k, v in rewards.items()}
        try:
            result.reward, result.reward_key = _pick_reward(result.rewards, reward_key)
        except ValueError as exc:
            result.ok = False
            result.error = str(exc)
        for step in getattr(trial_result, "step_results", None) or []:
            step_rewards = dict(
                getattr(getattr(step, "verifier_result", None), "rewards", None) or {}
            )
            result.step_results.append(
                HarborStepResult(
                    name=getattr(step, "name", "") or "",
                    rewards={k: float(v) for k, v in step_rewards.items()},
                )
            )
        info = getattr(trial_result, "exception_info", None)
        if info is not None and result.error is None:
            result.ok = False
            result.exception_type = getattr(info, "exception_type", None)
            result.error = str(getattr(info, "exception_message", ""))[:600]

    # --- capture --------------------------------------------------------
    try:
        # `include_messages` is what puts the assistant's own output in the result.
        # Only the response side is kept downstream (see `turns_from_document`), so
        # the payload grows by the completion text, not by the whole conversation.
        document = export_session(session, include_messages=True)
        stats = document.get("stats", {})
        result.n_turns = stats.get("n_turns", 0)
        result.n_roots = stats.get("n_roots", 0)
        result.n_trainable_tokens = stats.get("n_trainable_tokens", 0)
        result.multi_turn = result.n_turns > result.n_roots
        result.findings = [
            f for f in document.get("validation", []) if not f.startswith("[INFO]")
        ]

        trial_dir = _trial_dir(trial_result, trials_dir, trial_name)
        # Not every harness writes ATIF. Three of the sixteen (hermes, openclaw, pi) emit no
        # `trajectory.json`, which left them with no independent check at all. pi does record the
        # same information in its own session log, so `load_trace` falls back to that; hermes writes
        # a zero-byte file and openclaw only echoes back its config, so for those two there is
        # genuinely nothing to compare against and `none` is the honest answer.
        atif, trace_source = load_trace(trial_dir) if trial_dir else (None, "")
        report = reconcile(document, atif)
        result.atif = "none" if atif is None else ("match" if report.ok else "MISMATCH")
        result.trace_source = trace_source
        result.findings += [
            str(f) for f in report.findings if not str(f).startswith("[INFO]")
        ]

        # Calls the harness's own trace does not count as agent steps are auxiliary; drop them so
        # they cannot be credited with the reward earned by solving the task.
        if report.aux_node_ids:
            aux = set(report.aux_node_ids)
            for sequence in document["sequences"]:
                if sequence["role"] == "agent" and set(sequence["node_ids"]) <= aux:
                    sequence["role"] = "auxiliary"

        result.turns = turns_from_document(document)
        # Every conversation, not only the trainable ones: an auxiliary call that
        # went wrong is exactly what someone reading a bad rollout needs to see.
        result.conversations = conversations_from_document(document)
        if not report.ok:
            result.ok = False
            result.error = result.error or "capture failed validation"

        # A FATAL from the document's own validation means the capture is unusable, and until now it
        # was recorded in `findings` without touching `ok`. Only the trace-reconciliation report
        # could set `ok`, so a rollout that reached the verifier but produced NO model calls came
        # back `ok=True` with zero trainable tokens, and reconciliation agreed because both sides
        # were empty. One such rollout even carried reward=1.0, which is the worst shape available:
        # a trainer filtering on `ok` would accept a row with nothing in it and a positive reward.
        fatal = [f for f in result.findings if f.startswith("[FATAL")]
        if fatal:
            result.ok = False
            result.error = result.error or fatal[0]
    except Exception as exc:  # noqa: BLE001
        result.ok = False
        result.error = (
            result.error or f"capture export failed: {type(exc).__name__}: {exc}"
        )
    finally:
        registry.delete(session.session_id)

    # A rollout that produced no reward is not a zero: the verifier never ran. Keeping the two
    # distinct is what stops a dead sandbox being scored as a wrong answer.
    if result.ok and result.reward is None and trial_result is not None:
        result.findings.append(
            "[WARN] ungraded: the verifier produced no reward for this trial"
        )
    return result


def _trial_dir(
    trial_result: Any, trials_dir: Path | str, trial_name: str
) -> Path | None:
    """Where Harbor wrote this trial's artifacts, including its trajectory."""
    uri = getattr(trial_result, "trial_uri", None) if trial_result is not None else None
    if uri:
        return Path(str(uri).replace("file://", ""))
    candidate = Path(str(trials_dir)) / trial_name
    return candidate if candidate.is_dir() else None


__all__ = ["run_rollout", "build_trial_config", "HarborRolloutResult", "HarborTurn"]
