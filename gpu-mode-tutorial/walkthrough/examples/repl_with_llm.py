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
    pip install transformers torch accelerate

Usage:
    python examples/repl_with_llm.py
"""
from __future__ import annotations

import os
from huggingface_hub import InferenceClient

from repl_env import REPLEnv
from repl_env.prompts import (
    RLM_SYSTEM_PROMPT,
    build_initial_prompt,
    extract_code_blocks,
    format_observation,
)


def create_qwen_llm():
    """Create an LLM function using the smallest Qwen instruct model."""
    #from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
    print(f"Loading model: {model_name}")

    client = InferenceClient(
        model=model_name,
        token=HF_TOKEN,
    )

    # Disable thinking mode for Qwen3
    enable_thinking = False

    def llm_fn(messages: list[dict]) -> str:
        """Generate response using Qwen model."""
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=2048,  # Inrecreased for longer code responses
            temperature=0.7,
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
    # Use the unified REPLEnv API (local mode)
    # Note: env max_iterations should be higher than LLM loop iterations
    # because each code block counts as one env iteration
    with REPLEnv() as env:
        result = env.reset(
            context=context,
            task_prompt=task_prompt,
            max_iterations=max_iterations * 5,  # Allow ~5 code blocks per LLM turn
        )
        obs = result.observation

        # Build initial messages using prompts from repl_env
        initial_user_prompt = build_initial_prompt(
            task_prompt=task_prompt,
            context_length=obs.context_length,
            context_preview=obs.context_preview,
            variables=obs.available_variables,
        )

        messages = [
            {"role": "system", "content": RLM_SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_prompt},
        ]

        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Get LLM response
            response = llm_fn(messages)

            if verbose:
                print(f"LLM Response:\n{response[:500]}{'...' if len(response) > 500 else ''}")

            # Extract and execute code blocks
            code_blocks = extract_code_blocks(response)

            if not code_blocks:
                # No code, ask LLM to provide code
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide Python code in ```python``` blocks to solve the task.",
                    }
                )
                continue

            # Execute each code block
            for code in code_blocks:
                if verbose:
                    print(f"\nExecuting:\n{code[:200]}{'...' if len(code) > 200 else ''}")

                result = env.execute(code)
                obs = result.observation

                if verbose:
                    print(f"Success: {obs.result.success}")
                    print(f"Env iteration: {obs.iteration}/{obs.max_iterations}")
                    if obs.result.stdout:
                        print(f"Output: {obs.result.stdout[:200]}")

                if result.done:
                    final_answer = env.state().final_answer
                    if verbose:
                        # Check if done due to FINAL() or max iterations
                        if final_answer:
                            print(f"\n=== Final Answer: {final_answer} ===")
                        else:
                            print(f"\n=== Environment terminated (iteration {obs.iteration}/{obs.max_iterations}) ===")
                    return final_answer

            # Format observation for next iteration
            obs_text = format_observation(obs)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs_text})

        # LLM loop max iterations reached (this is separate from env max_iterations)
        if verbose:
            print(f"\nLLM loop max iterations ({max_iterations}) reached")
            print("Note: Each code block counts as one env iteration.")
            print(f"Final env state: iteration {obs.iteration}/{obs.max_iterations}")
        return None


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
