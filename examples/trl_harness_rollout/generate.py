"""Seam 1 (`generate`) implementations.

`FakeGenerate` runs the whole harness dynamic with NO GPU and NO model (a scripted "model" that drives
the ReAct agent deterministically). `VLLMGenerate` is the real path: it generates with vLLM and
captures the engine's exact token_ids + logprobs.

`VLLMGenerate` does naive tokenization here so the example runs. In a real trainer integration this is
where the prefix-preserving, token-in/token-out tokenization (TITO) lives, on the trainer side. See
`trl_adapter.py`.
"""

from __future__ import annotations

import re
import threading
from typing import Any


# ── No-GPU scripted model: drives the ReAct harness deterministically ──
class FakeGenerate:
    """Lets the full harness dynamic (interception, multi-turn, verify) run with no GPU and no model.

    Turn logic mirrors the ReAct protocol: if the last message is a calc result, answer with it,
    otherwise emit ACTION: calc(<expression from the question>).
    """

    def generate(self, *, rollout_id, turn, messages, tools, sampling) -> str:
        last = messages[-1]["content"] if messages else ""
        m = re.search(r"calc result:\s*(-?\d+)", last)
        if m:
            return f"ANSWER: {m.group(1)}"
        question = next(
            (
                mm["content"]
                for mm in messages
                if mm.get("role") == "user" and "What is" in mm["content"]
            ),
            "",
        )
        e = re.search(r"What is (.+?)\?", question)
        return f"ACTION: calc({e.group(1) if e else '0'})"


# ── Real path: vLLM generation + token_ids/logprobs capture ──
class VLLMGenerate:
    def __init__(self, *, vllm_base_url: str, model: str, api_key: str = "token"):
        from transformers import AutoTokenizer

        self._url = vllm_base_url.rstrip("/")
        self._model = model
        self._key = api_key
        self._tok = AutoTokenizer.from_pretrained(model)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self.captures: dict[tuple[str, int], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def generate(self, *, rollout_id, turn, messages, tools, sampling) -> str:
        import requests

        # NAIVE tokenization. The trainer replaces this with the TITO-safe, prefix-preserving path.
        encoded = self._tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        prompt_ids = _to_id_list(encoded)
        r = requests.post(
            f"{self._url}/v1/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self._model,
                "prompt": prompt_ids,
                "max_tokens": sampling.get("max_tokens", 256),
                "temperature": sampling.get("temperature", 1.0),
                "logprobs": 0,
                "return_token_ids": True,  # real engine ids: the capture a TITO step would stitch
            },
            timeout=600,
        )
        r.raise_for_status()
        choice = r.json()["choices"][0]
        with self._lock:
            self.captures[(rollout_id, turn)] = {
                "prompt_ids": prompt_ids,
                "completion_ids": choice["token_ids"],
                "logprobs": choice["logprobs"]["token_logprobs"],
                "text": choice.get("text", ""),
            }
        return choice.get("text", "")

    def capture_summary(self) -> str:
        with self._lock:
            n = len(self.captures)
            toks = sum(len(c["completion_ids"]) for c in self.captures.values())
        return f"{n} captured turns, {toks} real completion tokens (with logprobs)"

    def dump(self, path: str) -> int:
        import json

        with self._lock:
            records = [
                {"rollout_id": rid, "turn": t, **cap}
                for (rid, t), cap in sorted(self.captures.items())
            ]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        return len(records)


def _to_id_list(encoded: Any) -> list[int]:
    """Normalize apply_chat_template output to a flat list[int] across transformers versions."""
    out = encoded
    if hasattr(out, "input_ids"):  # BatchEncoding
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # tensor
        out = out.tolist()
    if out and isinstance(out[0], list):  # batched [[...]] -> [...]
        out = out[0]
    return [int(t) for t in out]
