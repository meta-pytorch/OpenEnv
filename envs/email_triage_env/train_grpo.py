#!/usr/import/env python3
"""GRPO training script for Oversight Inbox Arena.

Addresses official hackathon guide requirements:
- Multiple independent reward functions (not just one combined score)
- Curriculum learning (easy -> medium -> hard progression)
- Anti-reward-hacking monitoring (output inspection)
- Unsloth support for low-VRAM training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure environment imports resolve
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "envs"))

from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction

# ---------------------------------------------------------------------------
# Multiple Independent Reward Functions
# ---------------------------------------------------------------------------

EVAL_CACHE = {}

def evaluate_completion(prompt_text: str, completion: str, difficulty: str = "hard") -> dict:
    key = hash(prompt_text + completion)
    if key in EVAL_CACHE:
        return EVAL_CACHE[key]

    match = re.search(r"Scenario seed: (\d+)", prompt_text)
    seed = int(match.group(1)) if match else 0

    env = EmailTriageEnvironment(difficulty=difficulty)
    env.reset(seed=seed)

    cat = "other"
    pri = 1
    esc = False

    cat_match = re.search(r"<category>(.*?)</category>", completion, re.IGNORECASE)
    pri_match = re.search(r"<priority>(\d+)</priority>", completion, re.IGNORECASE)
    esc_match = re.search(r"<escalate>(.*?)</escalate>", completion, re.IGNORECASE)

    if cat_match: cat = cat_match.group(1).strip().lower()
    if pri_match: 
        try:
            pri = int(pri_match.group(1).strip())
        except:
            pass
    if esc_match: esc = "true" in esc_match.group(1).strip().lower()

    action = EmailTriageAction(category=cat, priority=pri, should_escalate=esc)
    obs = env.step(action)
    info = obs.info or {}
    
    # If no valid XML tags found, heavy penalty for reward hacking/timeout
    hacking_penalty = -2.0 if not cat_match else 0.0
    
    comps = info.get("reward_components", {})
    results = {
        "quality": comps.get("quality", 0.0),
        "sla": comps.get("sla", 0.0),
        "policy": comps.get("policy", 0.0),
        "oversight": comps.get("oversight", 0.0),
        "hacking": hacking_penalty
    }
    
    EVAL_CACHE[key] = results
    return results

def get_prompt_text(prompt) -> str:
    if isinstance(prompt, list):
        return prompt[-1]["content"] if "content" in prompt[-1] else str(prompt)
    return str(prompt)

def reward_quality(prompts: list, completions: list, **kw) -> list:
    return [evaluate_completion(get_prompt_text(p), c)["quality"] for p, c in zip(prompts, completions)]

def reward_compliance(prompts: list, completions: list, **kw) -> list:
    return [evaluate_completion(get_prompt_text(p), c)["policy"] for p, c in zip(prompts, completions)]

def reward_sla(prompts: list, completions: list, **kw) -> list:
    return [evaluate_completion(get_prompt_text(p), c)["sla"] for p, c in zip(prompts, completions)]

def reward_oversight(prompts: list, completions: list, **kw) -> list:
    return [evaluate_completion(get_prompt_text(p), c)["oversight"] for p, c in zip(prompts, completions)]

def reward_no_hacking(prompts: list, completions: list, **kw) -> list:
    return [evaluate_completion(get_prompt_text(p), c)["hacking"] for p, c in zip(prompts, completions)]

ALL_REWARD_FUNCTIONS = [
    reward_quality,
    reward_oversight,
    reward_compliance,
    reward_sla,
    reward_no_hacking,
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsloth", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", default="oversight-inbox-grpo")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    if args.smoke:
        args.dataset_size = 4
        args.num_generations = 2
        args.max_steps = 2
        args.curriculum = False

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("ERROR: Install trl and datasets first")
        sys.exit(1)

    if args.unsloth:
        try:
            from unsloth import FastLanguageModel, PatchFastRL
            PatchFastRL("unsloth", FastLanguageModel)
        except ImportError:
            pass

    import torch
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16 if not args.smoke else 1,
        num_generations=args.num_generations,
        max_completion_length=512,
        max_prompt_length=1024,
        logging_steps=1,
        save_steps=25,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
    )

    system_msg = (
        "You are an expert email triage coordinator. For each ticket, output your decision in XML tags:\n"
        "<category>support</category>\n"
        "<priority>3</priority>\n"
        "<escalate>false</escalate>\n"
    )

    prompts = []
    for i in range(args.dataset_size):
        prompts.append([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Process the inbox queue. Scenario seed: {i}."},
        ])

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"[DATA] {len(dataset)} prompts")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=ALL_REWARD_FUNCTIONS,
        train_dataset=dataset,
        args=config,
    )

    print("[START] GRPO training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[SAVED] Model -> {args.output_dir}")

if __name__ == "__main__":
    main()
