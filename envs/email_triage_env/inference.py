from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import requests
from openai import OpenAI


BENCHMARK = "email_triage_env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY") or GROQ_API_KEY or HF_TOKEN or OPENAI_API_KEY
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

IMAGE_NAME = LOCAL_IMAGE_NAME or "email-triage-env-openenv:latest"
TEMPERATURE = 0.0
MAX_TOKENS = 200
PORT = 8012
CONTAINER_NAME = "email-triage-inference-run"

TASKS = [("easy", 11), ("medium", 22), ("hard", 33)]

SYSTEM_PROMPT = (
    "You are an email triage assistant. Return only compact JSON with keys "
    "category, priority, should_escalate. category must be one of "
    "billing/support/spam/urgent/marketing/other; priority must be int 1-5; "
    "should_escalate must be true or false."
)


@dataclass
class ParsedAction:
    category: str
    priority: int
    should_escalate: bool


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = out.strip("`")
        if out.startswith("json"):
            out = out[4:]
    return out.strip()


def _heuristic_action(subject: str, body: str) -> ParsedAction:
    msg = f"{subject} {body}".lower()
    if any(k in msg for k in ["outage", "incident", "critical", "urgent", "production"]):
        return ParsedAction("urgent", 5, True)
    if any(k in msg for k in ["prize", "click", "offer", "winner", "reward"]):
        return ParsedAction("spam", 1, False)
    if any(k in msg for k in ["invoice", "billing", "payment", "refund", "charge"]):
        return ParsedAction("billing", 3, False)
    if any(k in msg for k in ["newsletter", "campaign", "promo", "partnership"]):
        return ParsedAction("marketing", 2, False)
    if any(k in msg for k in ["support", "error", "issue", "login", "bug"]):
        return ParsedAction("support", 3, False)
    return ParsedAction("other", 2, False)


def _parse_model_action(text: str, subject: str, body: str) -> ParsedAction:
    cleaned = _strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        category = str(payload.get("category", "other")).lower().strip()
        if category not in {"billing", "support", "spam", "urgent", "marketing", "other"}:
            category = "other"
        priority = int(payload.get("priority", 2))
        priority = max(1, min(5, priority))
        should_escalate = bool(payload.get("should_escalate", False))
        return ParsedAction(category, priority, should_escalate)
    except Exception:
        return _heuristic_action(subject, body)


def _build_openai_client() -> Optional[OpenAI]:
    if not API_KEY:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _query_model(
    client: Optional[OpenAI],
    subject: str,
    body_snippet: str,
    sender_domain: str,
    task: str,
) -> ParsedAction:
    if client is None:
        return _heuristic_action(subject, body_snippet)

    user_prompt = (
        f"Task={task}. Sender domain={sender_domain}. Subject={subject}. "
        f"Body snippet={body_snippet}. Return JSON only."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return _parse_model_action(content, subject, body_snippet)
    except Exception:
        return _heuristic_action(subject, body_snippet)


def _docker_cleanup() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, text=True)


def _start_container() -> None:
    _docker_cleanup()
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"{PORT}:8000",
            IMAGE_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _wait_for_health(base_url: str, timeout_s: float = 45.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("Environment did not become healthy in time")


def _run_task(base_url: str, client: Optional[OpenAI], task_name: str, seed: int) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    reset_payload = {"difficulty": task_name, "seed": seed}
    reset_result = requests.post(f"{base_url}/reset", json=reset_payload, timeout=15).json()
    obs = reset_result["observation"]

    parsed = _query_model(client, obs["subject"], obs["body_snippet"], obs["sender_domain"], task_name)
    action = {
        "action": {
            "category": parsed.category,
            "priority": parsed.priority,
            "should_escalate": parsed.should_escalate,
        }
    }

    rewards: List[float] = []
    error: Optional[str] = None
    try:
        step_result = requests.post(f"{base_url}/step", json=action, timeout=15).json()
        reward = float(step_result.get("reward") or 0.0)
        done = bool(step_result.get("done", False))
    except Exception as exc:
        reward = 0.0
        done = True
        error = str(exc).replace(" ", "_")

    rewards.append(reward)
    action_repr = f"{parsed.category}|{parsed.priority}|{str(parsed.should_escalate).lower()}"
    log_step(step=1, action=action_repr, reward=reward, done=done, error=error)

    score = sum(rewards)
    log_end(success=score > 0.0, steps=len(rewards), score=score, rewards=rewards)
    return score


def main() -> None:
    client = _build_openai_client()
    base_url = f"http://127.0.0.1:{PORT}"

    _start_container()
    try:
        _wait_for_health(base_url)
        scores = []
        for task_name, seed in TASKS:
            scores.append(_run_task(base_url, client, task_name, seed))
        overall = sum(scores) / len(scores)
        print(f"FINAL_AVG_SCORE={overall:.3f}", flush=True)
    finally:
        _docker_cleanup()


if __name__ == "__main__":
    main()
