#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment with LLM Integration.

This example demonstrates how to use the REPL environment with an actual LLM,
implementing the Recursive Language Model (RLM) paradigm:

1. LLM generates Python code to solve a task
2. Code is executed in the sandboxed REPL
3. LLM sees the output and generates more code
4. Process repeats until FINAL() is called

This is similar to the MIT RLM implementation but uses OpenEnv's repl_env.

Requirements:
    pip install huggingface_hub

Usage:
    python examples/repl_with_llm.py
"""

from __future__ import annotations

import os
from huggingface_hub import InferenceClient

from repl_env import LocalRLMRunner, RLM_SYSTEM_PROMPT


def create_qwen_llm():
    """Create an LLM function using a small Qwen model."""
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    model_name = os.environ.get("REPL_LLM_MODEL", "Qwen/Qwen3.5-9B")
    print(f"Loading model: {model_name}")

    client = InferenceClient(
        model=model_name,
        token=HF_TOKEN,
    )

    def llm_fn(messages: list[dict], model: str | None = None) -> str:
        """Generate response using Qwen model."""
        response = client.chat.completions.create(
            model=model or model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            # Disable thinking mode — the RLM loop is the reasoning mechanism
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content

    return llm_fn


def run_rlm_loop(
    llm_fn,
    context: str,
    task_prompt: str,
    max_iterations: int = 30,
    verbose: bool = True,
):
    """
    Run the RLM loop with an LLM and the REPL environment.

    Args:
        llm_fn: Function that takes messages and returns LLM response
        context: The context/data to process
        task_prompt: The task to accomplish
        max_iterations: Maximum REPL iterations
        verbose: Print progress

    Returns:
        The final answer string
    """
    runner = LocalRLMRunner(
        llm_fn,
        system_prompt=RLM_SYSTEM_PROMPT,
        max_iterations=max_iterations,
        max_depth=3,
        verbose=verbose,
    )
    return runner.run(context, task_prompt).final_answer


def main():
    """Run example with Qwen model."""
    print("=" * 60)
    print("REPL Environment with LLM Integration (Qwen)")
    print("=" * 60)

    # Create the LLM function
    llm_fn = create_qwen_llm()

    # Example task
    context = """
    The quick brown fox jumps over the lazy dog.
    This is a sample text for testing the REPL environment.
    It contains multiple sentences that we can analyze.
    The RLM paradigm allows models to process data programmatically.
    """

    task = "Count the total number of words in the context"

    print(f"\nTask: {task}")
    print(f"Context: {context[:100]}...")

    result = run_rlm_loop(
        llm_fn=llm_fn,
        context=context,
        task_prompt=task,
        max_iterations=10,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"Final Result: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
