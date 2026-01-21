#!/usr/bin/env python3
"""
Simple REPL + Oolong example with recursive LLM calls (RLM paradigm).

Demonstrates the unified REPLEnv API that works with both remote servers
and local execution using the same interface.

Usage:
    # Run against remote server
    python examples/repl_oolong_simple.py

    # Run locally (set SPACE_URL = None in the script)
    python examples/repl_oolong_simple.py
"""
from __future__ import annotations

import os

from datasets import load_dataset
from huggingface_hub import InferenceClient

# HuggingFace token for Inference API
HF_TOKEN = os.environ.get("HF_TOKEN")


from repl_env import REPLEnv
from repl_env.prompts import (
    RLM_SYSTEM_PROMPT_QWEN,  # Use Qwen version (with cost warning)
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
    extract_code_blocks,
    format_observations,
)

# ============== CONFIGURATION ==============
# Set to None to run locally, or a URL to connect to remote Space
SPACE_URL = "https://sergiopaniego-repl.hf.space"
MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
DATASET_SUBSET = "toy_dnd"
DATASET_SPLIT = "validation"
EXAMPLE_INDEX = 0
MAX_ITERATIONS = 30  # Paper uses 30
# ===========================================


def main():
    print("=" * 60)
    print("REPL + Oolong with Recursive LLM Calls (RLM)")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset example {EXAMPLE_INDEX}...")
    dataset = load_dataset("oolongbench/oolong-real", DATASET_SUBSET, split=DATASET_SPLIT)
    example = dataset[EXAMPLE_INDEX]

    context = example["context_window_text"]
    question = example["question"]
    expected = str(example["answer"])

    print(f"Question: {question}")
    print(f"Expected answer: {expected}")
    print(f"Context length: {len(context):,} chars")

    # Load model for the outer loop (agent)
    client = InferenceClient(
        model=MODEL_NAME,
        token=HF_TOKEN,
    )

    def llm_chat(messages: list[dict]) -> str:
        """
        LLM function for chat-style messages (outer loop),
        using HF Inference Providers.
        """
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=2048,  # Increased for longer code responses
            temperature=0.7,
        )
        return response.choices[0].message.content

    # Build task prompt (just the question, as per official RLM)
    task_prompt = question

    # Create environment - unified API for both local and remote!
    if SPACE_URL:
        print(f"\nConnecting to: {SPACE_URL}")
        env = REPLEnv(base_url=SPACE_URL)
    else:
        print("\nRunning locally")
        # For local mode, provide LLM functions for llm_query/llm_query_batched support
        def local_llm_query(prompt: str) -> str:
            return llm_chat([{"role": "user", "content": prompt}])

        def local_llm_batch(prompts: list[str]) -> list[str]:
            return [local_llm_query(p) for p in prompts]

        env = REPLEnv(llm_query_fn=local_llm_query, llm_batch_fn=local_llm_batch)

    # Reset environment - same API for both local and remote
    # Pass hf_token so the server uses our token for llm_query/llm_query_batched
    result = env.reset(
        context=context,
        task_prompt=task_prompt,
        max_iterations=MAX_ITERATIONS,
        hf_token=HF_TOKEN,  # Server will use this token for sub-LLM calls
    )
    obs = result.observation

    print(f"Context loaded: {obs.context_length:,} chars")
    print(f"Available variables: {obs.available_variables}")

    # Build initial messages (official RLM style):
    # 1. System prompt
    # 2. Assistant message with context metadata
    # 3. User prompt with safeguard
    query_metadata = QueryMetadata(
        context_lengths=[obs.context_length],
        context_total_length=obs.context_length,
        context_type="str",
    )

    messages = build_rlm_system_prompt(RLM_SYSTEM_PROMPT_QWEN, query_metadata)
    messages.append(build_user_prompt(root_prompt=task_prompt, iteration=0))

    # RLM loop
    final_answer = None
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {i} ---")

        response = llm_chat(messages)
        print(f"LLM: {response[:400]}{'...' if len(response) > 400 else ''}")

        code_blocks = extract_code_blocks(response)
        code_block_observations = []

        if not code_blocks:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please provide code in ```repl``` blocks."})
            continue

        for code in code_blocks:
            print(f"\nExecuting:\n{code[:300]}{'...' if len(code) > 300 else ''}")

            # Execute code - same API for both local and remote!
            result = env.execute(code)
            obs = result.observation
            code_block_observations.append(obs)

            print(f"Success: {obs.result.success}")
            print(f"Env iteration: {obs.iteration}/{obs.max_iterations}")
            if obs.result.stdout:
                print(f"Output: {obs.result.stdout[:300]}{'...' if len(obs.result.stdout) > 300 else ''}")
            if obs.result.stderr:
                print(f"Stderr: {obs.result.stderr[:200]}")

            if result.done:
                state = env.state()
                final_answer = state.final_answer
                if final_answer:
                    print(f"\n=== FINAL answer detected ===")
                else:
                    print(f"\n=== Environment terminated (max iterations) ===")
                break

        if result.done:
            break  # Exit outer loop when env is done (with or without answer)

        # Add assistant response and observation + next user prompt
        messages.append({"role": "assistant", "content": response})
        observation_text = format_observations(code_block_observations)
        next_prompt = build_user_prompt(root_prompt=task_prompt, iteration=i)
        messages.append({"role": "user", "content": observation_text + "\n\n" + next_prompt["content"]})

    # Cleanup
    env.close()

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"Got:      {final_answer}")

    if final_answer and str(final_answer).strip().lower() == expected.strip().lower():
        print("✓ CORRECT!")
    else:
        print("✗ INCORRECT")


if __name__ == "__main__":
    main()
