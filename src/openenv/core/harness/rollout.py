"""Minimal contract for training an agentic harness (OpenEnv side).

This module is the small, framework-neutral seam between an OpenEnv harness (an agent that owns its
loop) and an online RL trainer. It defines the interfaces only. The trainer (for example TRL) provides
the rollout worker that drives a harness through these, generates, and emits the rollout. See
`interception.InterceptionServer` for the transport.

The two seams:
  Seam 1 (the trainer implements):
      generate(rollout_id, turn, messages, tools, sampling) -> completion_text
      The trainer generates with its inference engine and records token_ids + logprobs keyed by
      (rollout_id, turn). Returns the completion text so the worker can hand it back to the agent.
  Seam 2 (the trainer's worker emits, the trainer consumes):
      RolloutMessages{rollout_id, messages, reward}

Reward stays in the env (the session's `verify`). Tokenization (messages to tokens, the
prefix-preserving token-in/token-out work) is the trainer's job, behind the `generate` seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


# ── Seam 1: the generate API (the trainer implements this) ──
class GenerateAPI(Protocol):
    def generate(
        self,
        *,
        rollout_id: str,
        turn: int,
        messages: list[Message],
        tools: list | None,
        sampling: dict,
    ) -> str:
        """Generate one assistant turn. The trainer records token_ids + logprobs keyed by
        (rollout_id, turn) and returns the completion text."""
        ...


# ── Seam 2: what the worker emits (the trainer consumes; the trainer does messages -> tokens) ──
@dataclass
class RolloutMessages:
    rollout_id: str
    messages: list[Message]  # full transcript in order (user / assistant / tool)
    reward: float
    metrics: dict = field(default_factory=dict)


# ── The harness session (OpenEnv side): one agent session per rollout ──
class AgentSession(Protocol):
    def next_request(self) -> dict | None:
        """Intercepted agent LLM call ({messages, tools, request_id}), or None when the agent exits."""
        ...

    def deliver(self, intercept: dict, completion_text: str) -> None: ...
    def verify(
        self,
    ) -> Any: ...  # returns an object with .env_reward (reward stays in the env)
    def close(self) -> None: ...


class AgentSessionFactory(Protocol):
    def create(self, *, task: Any, rollout_id: str) -> AgentSession: ...
