#!/usr/bin/env python3
"""Solve a coding task with a hosted LLM via Hugging Face Inference.

This script mirrors ``textarena_wordle_inference.py`` but targets the Coding
environment. It launches the CodingEnv Docker image locally and asks an
OpenAI-compatible model served through Hugging Face's router to iteratively
produce Python code until the task is solved.

Prerequisites
-------------
1. Build the Coding environment Docker image::

       docker build \
           -f envs/coding_env/server/Dockerfile \
           -t coding-env:latest .

2. Set your Hugging Face token, or any other API key that is compatible with the OpenAI API:

       export HF_TOKEN=your_token_here
       export API_KEY=your_api_key_here

3. Run the script::

       python examples/coding_env_inference.py

The script keeps sending execution feedback to the model until it prints
``Result: 338350`` or reaches the configured step limit.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from openai import OpenAI

from coding_env import CodeAction, CodingEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b:novita"
MAX_STEPS = 5
VERBOSE = True

CODING_TASK = (
    "Write Python code that prints the sum of squares of the integers from 1 "
    "to 100 inclusive. The final line must be exactly `Result: <value>` with "
    "the correct number substituted."
)
EXPECTED_SUBSTRING = "Result: 338350"

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Respond with valid Python code that "
    "solves the user's task. Always wrap your final answer in a fenced code "
    "block starting with ```python. Provide a complete script that can be "
    "executed as-is, with no commentary outside the code block."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_python_code(text: str) -> str:
    """Extract the first Python code block from the model output."""

    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


def format_feedback(
    step: int,
    stdout: str,
    stderr: str,
    exit_code: int,
) -> str:
    """Generate feedback text describing the previous execution."""

    stdout_display = stdout if stdout.strip() else "<empty>"
    stderr_display = stderr if stderr.strip() else "<empty>"
    return (
        f"Execution feedback for step {step}:\n"
        f"exit_code={exit_code}\n"
        f"stdout:\n{stdout_display}\n"
        f"stderr:\n{stderr_display}\n"
        "If the task is not solved, return an improved Python script."
    )


def build_initial_prompt(task: str) -> str:
    """Construct the first user prompt for the coding task."""

    return (
        "You must write Python code to satisfy the following task. "
        "When executed, your script should behave exactly as described.\n\n"
        f"Task:\n{task}\n\n"
        "Reply with the full script in a single ```python code block."
    )


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------


def solve_coding_task(
    env: CodingEnv,
    client: OpenAI,
) -> Tuple[bool, List[str]]:
    """Iteratively ask the model for code until the task is solved."""

    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_prompt(CODING_TASK)},
    ]

    obs = env.reset().observation

    transcripts: List[str] = []

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            max_tokens=2048,
            temperature=0.2,
        )

        assistant_message = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": assistant_message})

        code = extract_python_code(assistant_message)

        if VERBOSE:
            print(f"\n🛠️  Step {step}: executing model-produced code")
            print(code)

        result = env.step(CodeAction(code=code))
        obs = result.observation

        transcripts.append(
            (
                f"Step {step} | exit_code={obs.exit_code}\nstdout:\n{obs.stdout}\nstderr:\n{obs.stderr}\n"
            )
        )

        if VERBOSE:
            print("   ▶ exit_code:", obs.exit_code)
            if obs.stdout:
                print("   ▶ stdout:\n" + obs.stdout)
            if obs.stderr:
                print("   ▶ stderr:\n" + obs.stderr)

        solved = obs.exit_code == 0 and EXPECTED_SUBSTRING in obs.stdout
        if solved:
            return True, transcripts

        history.append(
            {
                "role": "user",
                "content": format_feedback(
                    step,
                    obs.stdout,
                    obs.stderr,
                    obs.exit_code,
                ),
            }
        )

        # Keep conversation history compact to avoid exceeding context limits
        if len(history) > 20:
            history = [history[0]] + history[-19:]

    return False, transcripts


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "HF_TOKEN (or API_KEY) must be set to query the model."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = CodingEnv.from_docker_image(
        "coding-env:latest",
        ports={8000: 8000},
    )

    try:
        success, transcripts = solve_coding_task(env, client)
    finally:
        env.close()

    print(
        "\n✅ Session complete"
        if success
        else "\n⚠️ Session finished without solving the task"
    )
    print("--- Execution transcripts ---")
    for entry in transcripts:
        print(entry)


if __name__ == "__main__":
    main()
