# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RLM System Prompts and Parsing Utilities for the REPL Environment.

Based on the official RLM prompts from https://github.com/alexzhang13/rlm
with adaptations for OpenEnv's repl_env.

These prompts guide LLMs to use the REPL environment effectively:
- Explore context programmatically
- Use llm_query/llm_batch for recursive calls
- Provide final answers via FINAL() or FINAL_VAR() patterns

Parsing utilities help extract code blocks and format observations.
"""

import re
import textwrap
from typing import Any, List, Optional


# Main system prompt based on official RLM implementation
RLM_SYSTEM_PROMPT = textwrap.dedent("""
    You are tasked with answering a query with associated context. You can access,
    transform, and analyze this context interactively in a REPL environment that can
    recursively query sub-LLMs.

    The REPL environment is initialized with:
    1. A `context` variable containing the data you need to analyze.
    2. A `llm_query(prompt)` function to query a sub-LLM for semantic analysis.
    3. A `llm_batch(prompts)` function for concurrent LLM queries.
    4. An `answer` dict for storing your final answer.
    5. Standard library modules (re, json, math, collections, itertools, etc.)

    To execute Python code, wrap it in triple backticks:
    ```python
    # Example: explore the context
    print(f"Context has {len(context)} chars")
    chunks = context.split('\\n')
    print(f"Found {len(chunks)} lines")
    ```

    Key strategies:
    - First explore the context to understand its structure
    - For long contexts, chunk and analyze with llm_batch
    - Use variables as buffers to build up your answer
    - DO NOT redefine the `context` variable - it's already set!

    When you have the final answer, use ONE of these patterns:
    1. FINAL(your_answer) - call the FINAL function with your answer
    2. print(f'FINAL({your_answer})') - print the final answer
    3. answer['content'] = result; answer['ready'] = True - use the answer dict

    Think step by step. Explore first, then analyze, then provide your final answer.
""").strip()


# Shorter system prompt for smaller models
RLM_SYSTEM_PROMPT_COMPACT = textwrap.dedent("""
    You have access to a Python REPL with a `context` variable containing data to analyze.

    Available:
    - context: the data (DO NOT redefine it)
    - llm_query(prompt): query a sub-LLM
    - llm_batch(prompts): concurrent sub-LLM queries
    - FINAL(answer): call this to submit your final answer
    - Standard library (re, json, math, collections, etc.)

    Write Python in ```python``` blocks.
    When done, call: FINAL(your_answer)
""").strip()


# Initial user prompt template
USER_PROMPT_INITIAL = textwrap.dedent("""
    Task: {task_prompt}

    Context ({context_length} chars):
    {context_preview}{ellipsis}

    Available variables: {variables}

    Think step-by-step:
    1. First explore the context to understand its structure
    2. Analyze the data to solve the task
    3. Provide your final answer with FINAL(answer)

    Write Python code in ```python``` blocks.
""").strip()


# Continuation prompt after code execution
USER_PROMPT_CONTINUE = textwrap.dedent("""
    Code output:
    {output}
    {error_section}
    Variables: {variables}

    Continue solving the task. Write more code or provide FINAL(answer).
""").strip()


# First iteration safeguard (from official RLM)
FIRST_ITERATION_SAFEGUARD = (
    "You have not explored the context yet. "
    "First look through it to understand the data before providing a final answer."
)


def build_initial_prompt(
    task_prompt: str,
    context_length: int,
    context_preview: Optional[str],
    variables: List[str],
    preview_limit: int = 500,
) -> str:
    """Build the initial user prompt.

    Args:
        task_prompt: The task to accomplish
        context_length: Total length of the context
        context_preview: Preview of the context (first N chars)
        variables: List of available variable names
        preview_limit: Max chars before adding ellipsis

    Returns:
        Formatted initial prompt string
    """
    return USER_PROMPT_INITIAL.format(
        task_prompt=task_prompt,
        context_length=context_length,
        context_preview=context_preview or "(empty)",
        ellipsis="..." if context_length > preview_limit else "",
        variables=variables,
    )


def build_continuation_prompt(
    stdout: str,
    stderr: Optional[str],
    exception: Optional[str],
    success: bool,
    variables: List[str],
    max_error_length: int = 300,
) -> str:
    """Build the continuation prompt after code execution.

    Args:
        stdout: Standard output from execution
        stderr: Standard error from execution
        exception: Exception message if any
        success: Whether execution succeeded
        variables: List of available variable names
        max_error_length: Max length for error messages

    Returns:
        Formatted continuation prompt string
    """
    output = stdout.strip() if stdout else "(no output)"

    if success:
        error_section = ""
    else:
        error = stderr or exception or "Unknown error"
        if len(error) > max_error_length:
            error = error[:max_error_length] + "..."
        error_section = f"\nERROR: {error}\nFix the error. Remember: 'context' is already defined.\n"

    return USER_PROMPT_CONTINUE.format(
        output=output,
        error_section=error_section,
        variables=variables,
    )


def build_messages(
    system_prompt: str,
    task_prompt: str,
    context_length: int,
    context_preview: Optional[str],
    variables: List[str],
    compact: bool = False,
) -> List[dict]:
    """Build the initial message list for the LLM.

    Args:
        system_prompt: Custom system prompt (or use default)
        task_prompt: The task to accomplish
        context_length: Total context length
        context_preview: Preview of context
        variables: Available variables
        compact: Use compact prompt for smaller models

    Returns:
        List of message dicts for chat API
    """
    if system_prompt is None:
        system_prompt = RLM_SYSTEM_PROMPT_COMPACT if compact else RLM_SYSTEM_PROMPT

    user_prompt = build_initial_prompt(
        task_prompt=task_prompt,
        context_length=context_length,
        context_preview=context_preview,
        variables=variables,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# Parsing Utilities
# =============================================================================


def extract_code_blocks(text: str, language: str = "python") -> List[str]:
    """Extract code blocks from LLM response.

    Supports both ```python``` and ```repl``` style blocks (as used by official RLM).

    Args:
        text: The LLM response text
        language: Language identifier to match (default "python")

    Returns:
        List of code strings extracted from the response
    """
    # Match both the specified language and 'repl' (used by official RLM)
    patterns = [
        rf"```{language}\s*(.*?)```",
        r"```repl\s*(.*?)```",
    ]

    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        all_matches.extend(m.strip() for m in matches if m.strip())

    return all_matches


def format_observation(obs: Any) -> str:
    """Format a REPLObservation into a continuation prompt for the LLM.

    This is a convenience function that extracts fields from the observation
    and calls build_continuation_prompt.

    Args:
        obs: The REPLObservation from env.step()

    Returns:
        Formatted prompt string for the next LLM turn
    """
    return build_continuation_prompt(
        stdout=obs.result.stdout or "",
        stderr=obs.result.stderr,
        exception=obs.result.exception,
        success=obs.result.success,
        variables=obs.available_variables,
    )
