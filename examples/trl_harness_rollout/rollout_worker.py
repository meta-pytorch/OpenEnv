"""Harness rollout worker (the TRL side).

Drives an agentic harness (an agent that owns its loop) through the interception proxy, scores the
episode with the env's `verify()`, and emits a message-level rollout. It does NOT tokenize:
tokenization (the prefix-preserving token-in/token-out work) is the trainer's job, behind the
`generate` seam.

This worker lives on the trainer side. It builds on a small, framework-neutral contract
(`rollout.py`: `AgentSession`, `RolloutMessages`, `GenerateAPI`) and an interception transport
(`interception.py`). Both are vendored here to keep the example self-contained, and mirror OpenEnv's
minimal contract.
"""

from __future__ import annotations

import uuid
from typing import Any

from rollout import AgentSessionFactory, GenerateAPI, RolloutMessages


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
        last_messages: list[dict] = []
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
