"""Unit tests for the harness rollout worker. No GPU, no model, no HTTP.

Run: pytest test_rollout_worker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import FakeGenerate  # noqa: E402
from rollout_worker import HarnessRolloutWorker, RolloutMessages  # noqa: E402


# ── In-memory stubs (no HTTP) to test the worker contract ──
class _StubSession:
    def __init__(self, n_turns: int, reward: float):
        self._n, self._reward, self._turn = n_turns, reward, 0
        self._msgs = [{"role": "user", "content": "task"}]

    def next_request(self):
        if self._turn >= self._n:
            return None
        return {
            "messages": list(self._msgs),
            "tools": None,
            "request_id": f"r{self._turn}",
        }

    def deliver(self, intercept, completion_text):
        self._msgs.append({"role": "assistant", "content": completion_text})
        self._msgs.append({"role": "tool", "content": "obs"})
        self._turn += 1

    def verify(self):
        return type("V", (), {"env_reward": self._reward})()

    def close(self):
        pass


class _StubFactory:
    def __init__(self, n_turns: int, reward: float):
        self._n, self._reward = n_turns, reward

    def create(self, *, task, rollout_id):
        return _StubSession(self._n, self._reward)


class _StubGenerate:
    def generate(self, *, rollout_id, turn, messages, tools, sampling) -> str:
        return f"completion-{turn}"


def test_worker_emits_message_level_rollout():
    worker = HarnessRolloutWorker(
        session_factory=_StubFactory(n_turns=2, reward=1.0),
        generate_api=_StubGenerate(),
        tasks=[{"id": "t"}],
    )
    r = worker.produce({"id": "t"})
    assert isinstance(r, RolloutMessages)
    assert r.reward == 1.0
    assert r.metrics["turns"] == 2.0
    assert r.messages[-1] == {"role": "assistant", "content": "completion-1"}
    assert any(m["role"] == "assistant" for m in r.messages)


def test_worker_returns_none_if_agent_never_sampled():
    worker = HarnessRolloutWorker(
        session_factory=_StubFactory(n_turns=0, reward=0.0),
        generate_api=_StubGenerate(),
        tasks=[{"id": "t"}],
    )
    assert worker.produce({"id": "t"}) is None


def test_fake_mode_end_to_end_solves_all():
    """Integration test of the full fake-mode path: proxy + react agent (HTTP) + FakeGenerate.

    Skipped if localhost binding is not permitted (e.g. a restricted sandbox)."""
    import pytest

    from harness import TASKS, ProxyAgentSessionFactory
    from interception import InterceptionProxy

    proxy = InterceptionProxy()
    try:
        proxy.start()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"localhost bind not available: {e}")

    try:
        worker = HarnessRolloutWorker(
            session_factory=ProxyAgentSessionFactory(proxy),
            generate_api=FakeGenerate(),
            tasks=TASKS,
            max_turns=6,
        )
        rollouts = [worker.produce(t) for t in TASKS]
        assert all(r is not None for r in rollouts)
        assert all(
            r.reward == 1.0 for r in rollouts
        )  # fake generator solves deterministically
        assert all(
            r.metrics["turns"] >= 2 for r in rollouts
        )  # multi-turn (ACTION then ANSWER)
    finally:
        proxy.stop()


def test_fake_mode_concurrent_multiplexes_by_rollout_id():
    """The proxy's value is multiplexing many rollouts at once. Run all tasks concurrently."""
    from concurrent.futures import ThreadPoolExecutor

    import pytest

    from harness import TASKS, ProxyAgentSessionFactory
    from interception import InterceptionProxy

    proxy = InterceptionProxy()
    try:
        proxy.start()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"localhost bind not available: {e}")

    try:
        worker = HarnessRolloutWorker(
            session_factory=ProxyAgentSessionFactory(proxy),
            generate_api=FakeGenerate(),
            tasks=TASKS,
            max_turns=6,
        )
        with ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
            rollouts = [r for r in pool.map(worker.produce, TASKS) if r is not None]
        assert len(rollouts) == len(TASKS)
        assert all(r.reward == 1.0 for r in rollouts)
    finally:
        proxy.stop()


def test_fake_generate_drives_react_protocol():
    gen = FakeGenerate()
    # Turn 0: question present, no calc result yet -> emit an ACTION calc(...)
    first = gen.generate(
        rollout_id="x",
        turn=0,
        tools=None,
        sampling={},
        messages=[{"role": "user", "content": "What is 347 * 29?"}],
    )
    assert first == "ACTION: calc(347 * 29)"
    # Next turn: a calc result is present -> answer with it
    second = gen.generate(
        rollout_id="x",
        turn=1,
        tools=None,
        sampling={},
        messages=[
            {"role": "user", "content": "What is 347 * 29?"},
            {"role": "assistant", "content": "ACTION: calc(347 * 29)"},
            {"role": "user", "content": "calc result: 10063"},
        ],
    )
    assert second == "ANSWER: 10063"
