#!/usr/bin/env python3
"""
TB2 + Local LLM example (camel-ai/seta-rl-qwen3-8b).

Demonstrates using the TB2 environment with a locally running model
to solve terminal tasks from Terminal-Bench-2.

Model: https://huggingface.co/camel-ai/seta-rl-qwen3-8b
Tasks: https://github.com/laude-institute/terminal-bench-2

Usage:
    # Use default task (headless-terminal)
    python examples/tbench2_seta_example.py

    # Use a specific task
    TB2_TASK_ID=some-other-task python examples/tbench2_seta_example.py

    # Use a local server
    TB2_BASE_URL=http://localhost:8000 python examples/tbench2_seta_example.py

Requirements:
    pip install transformers torch accelerate
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tbench2_env import Tbench2Action, Tbench2Env

# ============== CONFIGURATION ==============
# HuggingFace Space URL for TB2 environment
# NOTE: The Space must have SETA tasks loaded (TB2_TASKS_DIR pointing to SETA dataset)
SPACE_URL = os.environ.get("TB2_BASE_URL", "https://openenv-tbench2.hf.space")

# Local model for generating commands (trained on SETA tasks)
MODEL_NAME = "camel-ai/seta-rl-qwen3-8b"

# Task ID from SETA dataset
# Available tasks: 1, 2, 3, 4, 5, ... (numeric IDs)
# Task 5 = shell redirection (difficulty: easy)
# See: https://huggingface.co/datasets/camel-ai/seta-env
TASK_ID = os.environ.get("TB2_TASK_ID", "5")

MAX_ITERATIONS = 15
# ===========================================


def load_local_model(model_name: str):
    """Load the SETA model locally."""
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


def generate_command(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
) -> str:
    """Generate a response from the local model."""
    # Format messages for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return response.strip()


def extract_commands(response: str) -> list[str]:
    """Extract shell commands from the model response.

    Looks for commands in code blocks (```bash or ```) or lines starting with $.
    """
    commands = []

    # Pattern 1: Code blocks with bash/shell
    code_block_pattern = r"```(?:bash|shell|sh)?\s*\n(.*?)```"
    for match in re.finditer(code_block_pattern, response, re.DOTALL):
        block = match.group(1).strip()
        # Split by newlines and filter empty lines
        for line in block.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Remove leading $ if present
                if line.startswith("$ "):
                    line = line[2:]
                commands.append(line)

    # Pattern 2: Lines starting with $ (if no code blocks found)
    if not commands:
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("$ "):
                commands.append(line[2:])

    return commands


def build_system_prompt() -> str:
    """Build system prompt for the SETA model."""
    return """You are a terminal agent that executes shell commands to complete tasks.

Given a task description, you should:
1. Analyze what needs to be done
2. Execute the necessary shell commands step by step
3. Verify your work

Respond with shell commands in ```bash``` code blocks.
Be precise and execute one logical step at a time.
After completing the task, say "DONE" to indicate completion."""


def build_user_prompt(instruction: str, last_output: str = "", iteration: int = 0) -> str:
    """Build user prompt with task and optional previous output."""
    if iteration == 0:
        return f"""Task:
{instruction}

Please execute the necessary commands to complete this task.
Start with the first step."""

    return f"""Previous command output:
{last_output}

Continue with the next step. If the task is complete, say "DONE"."""


def main():
    print("=" * 60)
    print("TB2 + Local Model Example")
    print("=" * 60)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Server: {SPACE_URL}")
    print(f"Task ID: {TASK_ID}")

    # Load model
    model, tokenizer = load_local_model(MODEL_NAME)
    print("Model loaded successfully!")

    # Connect to TB2 environment
    print(f"\nConnecting to TB2 server: {SPACE_URL}")
    env = Tbench2Env(base_url=SPACE_URL)

    # Reset environment with task
    print(f"Resetting with task_id: {TASK_ID}")
    result = env.reset(task_id=TASK_ID)

    # Get the REAL instruction from the task (not hardcoded)
    task_instruction = result.observation.instruction
    if not task_instruction:
        print("WARNING: No instruction found in task. Using generic prompt.")
        task_instruction = "Explore the environment and complete any required setup."

    print(f"\n{'='*40}")
    print("TASK INSTRUCTION (from server):")
    print("=" * 40)
    print(task_instruction[:500] if len(task_instruction) > 500 else task_instruction)
    if len(task_instruction) > 500:
        print("...")
    print("=" * 40)

    # Build initial messages with the REAL task instruction
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(task_instruction, iteration=0)},
    ]

    # Agent loop
    last_output = ""
    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*40}")
        print(f"Iteration {i}/{MAX_ITERATIONS}")
        print("=" * 40)

        # Generate response from model
        response = generate_command(model, tokenizer, messages)
        print(f"\nModel response:\n{response[:500]}{'...' if len(response) > 500 else ''}")

        # Check if done
        if "DONE" in response.upper() and i > 1:
            print("\n=== Agent indicates task is complete ===")
            break

        # Extract and execute commands
        commands = extract_commands(response)

        if not commands:
            print("No commands found in response. Asking for clarification...")
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "user", "content": "Please provide shell commands in ```bash``` code blocks."}
            )
            continue

        # Execute each command
        all_outputs = []
        for cmd in commands:
            print(f"\nExecuting: {cmd}")
            result = env.step(Tbench2Action(action_type="exec", command=cmd))

            output = result.observation.output
            success = result.observation.success
            error = result.observation.error

            print(f"Success: {success}")
            if output:
                print(f"Output: {output[:300]}{'...' if len(output) > 300 else ''}")
            if error:
                print(f"Error: {error[:200]}")

            all_outputs.append(f"$ {cmd}\n{output}" if output else f"$ {cmd}\n(no output)")

            if result.done:
                print("\n=== Environment indicates episode done ===")
                break

        if result.done:
            break

        # Update conversation
        last_output = "\n".join(all_outputs)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": build_user_prompt(task_instruction, last_output, i)})

    # Attempt formal evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    try:
        eval_result = env.step(Tbench2Action(action_type="evaluate"))
        print(f"Reward: {eval_result.reward}")
        print(f"Done: {eval_result.done}")

        if eval_result.reward == 1.0:
            print("\n✓ TASK COMPLETED SUCCESSFULLY!")
        else:
            print("\n✗ Task evaluation failed")
            if eval_result.observation.output:
                # Show last part of output (often contains error info)
                output = eval_result.observation.output
                print(f"\nEvaluation output (last 500 chars):")
                print(output[-500:] if len(output) > 500 else output)
    except Exception as e:
        print(f"Evaluation error: {e}")

    # Cleanup
    env.close()
    print("\nEnvironment closed.")


if __name__ == "__main__":
    main()
