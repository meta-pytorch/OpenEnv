#!/usr/bin/env python3
"""
Simple REPL + Oolong example with recursive LLM calls (RLM paradigm).

Uses LocalRLMRunner which handles both the outer loop (code generation)
and inner calls (llm_query/rlm_query) with a single chat function.

Usage:
    python examples/repl_oolong_simple.py
"""

from __future__ import annotations

import os

from datasets import load_dataset
from huggingface_hub import InferenceClient
from repl_env import LocalRLMRunner
from repl_env.prompts import RLM_SYSTEM_PROMPT_QWEN

# ============== CONFIGURATION ==============
MODEL_NAME = "Qwen/Qwen3.5-9B"
DATASET_SUBSET = "toy_dnd"
DATASET_SPLIT = "validation"
EXAMPLE_INDEX = 0
MAX_ITERATIONS = 30  # Paper uses 30
# ===========================================

HF_TOKEN = os.environ.get("HF_TOKEN")


def create_chat_fn():
    """Create the chat function with Qwen3.5 model card recommended params."""
    client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN, timeout=300)

    def chat_fn(messages: list[dict], model: str | None = None) -> str:
        response = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            # Qwen3.5 non-thinking mode for precise coding tasks (from model card)
            temperature=0.6,
            top_p=0.95,
            presence_penalty=0.0,
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                # Disable thinking mode — the RLM loop is the reasoning mechanism
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content

    return chat_fn


def main():
    print("=" * 60)
    print("REPL + Oolong with Recursive LLM Calls (RLM)")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset example {EXAMPLE_INDEX}...")
    dataset = load_dataset(
        "oolongbench/oolong-real", DATASET_SUBSET, split=DATASET_SPLIT
    )
    example = dataset[EXAMPLE_INDEX]

    context = example["context_window_text"]
    question = example["question"]
    expected = str(example["answer"])

    print(f"Question: {question}")
    print(f"Expected answer: {expected}")
    print(f"Context length: {len(context):,} chars")

    # Create LLM function — used for both outer loop and inner llm_query calls
    chat_fn = create_chat_fn()

    # Run the RLM loop
    runner = LocalRLMRunner(
        chat_fn,
        system_prompt=RLM_SYSTEM_PROMPT_QWEN,
        max_iterations=MAX_ITERATIONS,
        max_depth=2,
        verbose=True,
    )
    result = runner.run(context, question)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print(f"Got:      {result.final_answer}")
    print(f"Iterations: {result.iterations}")

    if (
        result.final_answer
        and str(result.final_answer).strip().lower() == expected.strip().lower()
    ):
        print("CORRECT!")
    else:
        print("INCORRECT")


if __name__ == "__main__":
    main()
