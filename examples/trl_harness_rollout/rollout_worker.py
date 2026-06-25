"""Harness rollout worker (OpenEnv side) for online RL training with TRL.

This is the OpenEnv half of the integration: it drives an agentic harness (an agent that owns its own
loop) through an interception proxy and emits a MESSAGE-level rollout (transcript + reward). It does
NOT tokenize. Tokenization (messages to tokens, the prefix-preserving token-in/token-out work) is the
training framework's job and plugs in behind the `generate` seam.

Two seams connect this to a trainer:
  Seam 1 (the trainer implements):
      generate(rollout_id, turn, messages, tools, sampling) -> completion_text
      The trainer generates with its inference engine and records token_ids + logprobs keyed by
      (rollout_id, turn). Returns the completion text so the worker can hand it back to the agent.
  Seam 2 (this worker emits, the trainer consumes):
      RolloutMessages{rollout_id, messages, reward}

The worker is harness-agnostic: all harness specificity lives in the AgentSession
(next_request / deliver / verify). Swapping a toy ReAct agent for a real coding agent does not change
this file.
"""

from __future__ import annotations

import uuid
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


# ── The worker ──
class HarnessRolloutWorker:
    """Drives a harness and emits message-level rollouts. Tokenization is the trainer's job."""

    def __init__(
        self,
        *,
        session_factory: AgentSessionFactory,
        generate_api: GenerateAPI,
        tasks: list[Any],
        max_turns: int = 0,  # 0 = until the agent exits
        sampling: dict | None = None,
    ):
        self._factory = session_factory
        self._gen = generate_api
        self._tasks = list(tasks)
        self._max_turns = max_turns
        self._sampling = sampling or {"temperature": 1.0, "max_tokens": 256}

    def produce(self, task: Any) -> RolloutMessages | None:
        rollout_id = (
            uuid.uuid4().hex
        )  # full uuid: no collision risk for long training runs
        session = self._factory.create(task=task, rollout_id=rollout_id)
        last_messages: list[Message] = []
        last_completion: str | None = None
        turn = 0
        try:
            while self._max_turns == 0 or turn < self._max_turns:
                intercept = session.next_request()
                if intercept is None:  # agent exited
                    break
                last_messages = intercept["messages"]
                completion = self._gen.generate(  # trainer generates + captures, keyed by (id, turn)
                    rollout_id=rollout_id,
                    turn=turn,
                    messages=last_messages,
                    tools=intercept.get("tools"),
                    sampling=self._sampling,
                )
                last_completion = completion
                session.deliver(intercept, completion)
                turn += 1

            if last_completion is None:  # agent never sampled
                return None
            reward = float(getattr(session.verify(), "env_reward", 0.0) or 0.0)
            transcript = list(last_messages) + [
                {"role": "assistant", "content": last_completion}
            ]
            return RolloutMessages(
                rollout_id=rollout_id,
                messages=transcript,
                reward=reward,
                metrics={"reward": reward, "turns": float(turn)},
            )
        finally:
            session.close()
