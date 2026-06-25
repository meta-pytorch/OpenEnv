"""A harness (an agent that owns its own loop) plus its OpenEnv session, task, and verifier.

The harness here is a small ReAct agent with a calculator tool, talking OpenAI over HTTP to the
interception proxy. It owns its loop: it decides when to call the tool and when to answer. This gives
the right dynamic (agent-owned loop, multi-turn, interception-gated) without a full coding agent. To
raise fidelity, swap this for a real harness (Pi / OpenCode over OpenEnv's InterceptionServer, PR #694)
in a follow-up: the rollout worker does not change.

Exposes `ProxyAgentSessionFactory`, which satisfies the worker's `AgentSessionFactory` protocol.
"""

from __future__ import annotations

import re
import threading
from typing import Any

import requests
from interception import InterceptionServer


# ── Task + verifier (the env): verifiable arithmetic ──
TASKS = [
    {"id": "t1", "prompt": "What is 347 * 29?", "answer": 10063},
    {"id": "t2", "prompt": "What is 9876 / 12?", "answer": 823},
    {"id": "t3", "prompt": "What is 1234 + 5678 - 999?", "answer": 5913},
    {"id": "t4", "prompt": "What is (84 + 16) * 47?", "answer": 4700},
]

_SYS = (
    "You solve an arithmetic question. You are bad at mental math, so you MUST use the calculator.\n"
    "To use it, output exactly one line and nothing else:\n"
    "ACTION: calc(<expression>)\n"
    "Only after you have seen a calc result, output exactly one line:\n"
    "ANSWER: <number>\n"
    "Never output ANSWER before using ACTION: calc(...) at least once."
)
_SAFE = re.compile(r"^[0-9+\-*/(). ]+$")


def calc(expr: str) -> str:
    expr = expr.strip()
    if not _SAFE.match(expr):
        return "error: invalid expression"
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (regex-restricted input)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        return str(val)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


# ── The agent: owns its loop, talks OpenAI over HTTP to the interception proxy ──
def react_agent(
    base_url: str, rollout_id: str, prompt: str, max_turns: int = 6
) -> None:
    url = f"{base_url}/rollout/{rollout_id}/v1/chat/completions"
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": prompt},
    ]
    used_calc = False
    for _ in range(max_turns):
        content = requests.post(url, json={"messages": messages}, timeout=600).json()
        content = content["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": content})
        m = re.search(
            r"ACTION:\s*calc\((.+)\)", content
        )  # greedy: capture up to the last ) on the line
        if m:
            used_calc = True
            messages.append(
                {"role": "user", "content": f"calc result: {calc(m.group(1))}"}
            )
            continue
        if "ANSWER:" in content:
            if used_calc:
                break
            messages.append(
                {"role": "user", "content": "Use ACTION: calc(...) first, then ANSWER."}
            )
            continue
        messages.append(
            {"role": "user", "content": "Output ACTION: calc(...) or ANSWER: <number>."}
        )
    requests.post(f"{base_url}/rollout/{rollout_id}/exit", json={}, timeout=30)


# ── OpenEnv session over the proxy (satisfies AgentSession) ──
class ProxyAgentSession:
    def __init__(self, proxy: InterceptionServer, rollout_id: str, q, answer: int):
        self._proxy, self._rid, self._q, self._answer = proxy, rollout_id, q, answer
        self._last = ""

    def next_request(self) -> dict | None:
        try:
            req_id = self._q.get(timeout=120)
        except Exception:
            return None
        return None if req_id is None else self._proxy.get_intercept(req_id)

    def deliver(self, intercept: dict, completion_text: str) -> None:
        self._last = completion_text
        self._proxy.deliver(intercept["request_id"], completion_text)

    def verify(self) -> Any:
        m = re.search(r"ANSWER:\s*(-?\d+)", self._last)
        ok = (
            m is not None and int(m.group(1)) == self._answer
        )  # reward stays in the env
        return type("V", (), {"env_reward": 1.0 if ok else 0.0})()

    def close(self) -> None:
        self._proxy.unregister_rollout(
            self._rid
        )  # drop the per-rollout queue, no leak on long runs


class ProxyAgentSessionFactory:
    def __init__(self, proxy: InterceptionServer):
        self._proxy = proxy

    def create(self, *, task: Any, rollout_id: str) -> ProxyAgentSession:
        q = self._proxy.register_rollout(rollout_id)
        threading.Thread(
            target=react_agent,
            args=(self._proxy.base_url, rollout_id, task["prompt"]),
            daemon=True,
        ).start()
        return ProxyAgentSession(self._proxy, rollout_id, q, task["answer"])
