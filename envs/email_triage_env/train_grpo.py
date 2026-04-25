#!/usr/bin/env python3
"""GRPO training script for Oversight Inbox Arena.

Designed to run on Google Colab Free Tier (T4 GPU, 15 GB VRAM).
Default model: Qwen/Qwen2-0.5B (500M params - fits easily on T4)
Larger option: Qwen/Qwen2-1.5B (requires Colab Pro)

Hackathon requirements:
- 5 independent reward functions (not one combined score)
- Anti-reward-hacking: penalizes missing XML structure
- Deterministic environments via seeding
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import gc
from typing import Any, Optional

# ── Memory fragmentation fix (MUST be before torch import) ──────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "envs"))

from email_triage_env.server.email_triage_environment import EmailTriageEnvironment
from email_triage_env.models import EmailTriageAction


# ── Reward evaluation cache (avoids re-running env for same prompt+completion)
_CACHE: dict = {}


def _text(obj: Any) -> str:
    """Safely extract string from str, list-of-dicts, or anything else."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        # Chat format: [{"role": "user", "content": "..."}]
        for item in reversed(obj):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return str(obj)
    return str(obj)


def _score(prompt: Any, completion: Any) -> dict:
    """Run one environment step and return reward components."""
    prompt_text = _text(prompt)
    completion_text = _text(completion)

    cache_key = hash(prompt_text[-100:] + completion_text[:200])
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    # Extract seed from prompt
    m = re.search(r"seed[:\s]+(\d+)", prompt_text, re.IGNORECASE)
    seed = int(m.group(1)) if m else 0

    # Parse XML tags from model output
    cat_m = re.search(r"<category>(.*?)</category>", completion_text, re.IGNORECASE)
    pri_m = re.search(r"<priority>(\d+)</priority>", completion_text, re.IGNORECASE)
    esc_m = re.search(r"<escalate>(true|false)</escalate>", completion_text, re.IGNORECASE)

    cat = cat_m.group(1).strip().lower() if cat_m else "other"
    pri = max(1, min(5, int(pri_m.group(1)))) if pri_m else 1
    esc = esc_m.group(1).lower() == "true" if esc_m else False

    # Penalty for not following format
    format_ok = cat_m is not None and pri_m is not None and esc_m is not None
    hacking_penalty = 0.0 if format_ok else -2.0

    try:
        env = EmailTriageEnvironment(difficulty="easy")
        env.reset(seed=seed)
        action = EmailTriageAction(category=cat, priority=pri, should_escalate=esc)
        obs = env.step(action)
        info = obs.info or {}
        comps = info.get("reward_components", {})
        result = {
            "quality":  float(comps.get("quality", 0.0)),
            "sla":      float(comps.get("sla", 0.0)),
            "policy":   float(comps.get("policy", 0.0)),
            "oversight":float(comps.get("oversight", 0.0)),
            "hacking":  hacking_penalty,
        }
        del env
    except Exception:
        result = {"quality": 0.0, "sla": 0.0, "policy": 0.0, "oversight": 0.0, "hacking": hacking_penalty}

    _CACHE[cache_key] = result
    return result


# ── 5 Independent Reward Functions ──────────────────────────────────────────

def reward_quality(prompts: list, completions: list, **kw) -> list:
    """Reward 1: Category + priority + escalation accuracy."""
    return [_score(p, c)["quality"] for p, c in zip(prompts, completions)]

def reward_sla(prompts: list, completions: list, **kw) -> list:
    """Reward 2: Resolved before SLA deadline."""
    return [_score(p, c)["sla"] for p, c in zip(prompts, completions)]

def reward_policy(prompts: list, completions: list, **kw) -> list:
    """Reward 3: Compliance with active policy rules."""
    return [_score(p, c)["policy"] for p, c in zip(prompts, completions)]

def reward_oversight(prompts: list, completions: list, **kw) -> list:
    """Reward 4: Specialist error correction / oversight quality."""
    return [_score(p, c)["oversight"] for p, c in zip(prompts, completions)]

def reward_format(prompts: list, completions: list, **kw) -> list:
    """Reward 5: Anti-hack - penalizes missing structured XML output."""
    return [_score(p, c)["hacking"] for p, c in zip(prompts, completions)]

ALL_REWARDS = [reward_quality, reward_sla, reward_policy, reward_oversight, reward_format]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO for Oversight Inbox Arena")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B",
                        help="Base model. Default: Qwen/Qwen2-0.5B (fits on free T4). "
                             "Use Qwen/Qwen2-1.5B for Colab Pro.")
    parser.add_argument("--output-dir", default="oversight-arena-grpo")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--dataset-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick 2-step smoke test to verify pipeline")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push trained model to HuggingFace Hub after training")
    parser.add_argument("--hub-repo", default="Rhushya/oversight-arena-model",
                        help="HuggingFace Hub repo ID to push to (e.g. username/model-name)")
    args = parser.parse_args()

    if args.smoke:
        args.max_steps = 2
        args.dataset_size = 4
        print("[SMOKE] Minimal run — just verifying pipeline works")

    # ── Imports ──────────────────────────────────────────────────────────────
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install trl datasets transformers accelerate torch")
        sys.exit(1)

    # ── GPU check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("WARNING: No GPU found. Training will be extremely slow.")
        print("In Colab: Runtime > Change Runtime Type > T4 GPU")

    gpu_free = 0
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        gpu_free = total - reserved
        print(f"[GPU] {torch.cuda.get_device_name(0)}: {total:.1f} GB total, {gpu_free:.1f} GB free")

    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[GPU] Precision: {'bf16' if is_bf16 else 'fp16'}")

    # ── Load model with Unsloth 4-bit if available ──────────────────────────
    model = args.model
    tokenizer = None

    try:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("unsloth", FastLanguageModel)
        print(f"[UNSLOTH] Loading {args.model} in 4-bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=256,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=8,
            gpu_memory_utilization=0.6,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=8,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"[UNSLOTH] 4-bit model loaded successfully")
    except ImportError:
        print("[INFO] Unsloth not found — using standard HuggingFace loading")
        print("[INFO] Install unsloth for 2x faster training on Colab T4")
    except Exception as e:
        print(f"[WARN] Unsloth loading failed: {e}")
        print("[INFO] Falling back to standard loading")
        model = args.model
        tokenizer = None

    # ── Clear any leftover CUDA memory ───────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_after = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
        print(f"[GPU] After model load: {free_after:.1f} GB free")

    # ── Training config ──────────────────────────────────────────────────────
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        optim="paged_adamw_8bit",       # Offloads optimizer states to RAM
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,              # Minimum for GRPO (needs contrast)
        max_completion_length=64,       # XML output is short
        max_prompt_length=128,          # Prompts are short
        logging_steps=1,
        save_steps=25,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        bf16=is_bf16,
        fp16=not is_bf16,
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    system_msg = (
        "You are an expert email triage coordinator. "
        "For each ticket, output your decision using exactly these XML tags:\n"
        "<category>billing</category>\n"
        "<priority>3</priority>\n"
        "<escalate>false</escalate>\n"
        "Valid categories: billing, support, spam, urgent, marketing, other\n"
        "Priority: 1 (lowest) to 5 (critical)"
    )

    prompts = [
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Triage the incoming email. seed: {i}"},
        ]
        for i in range(args.dataset_size)
    ]
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"[DATA] {len(dataset)} training prompts ready")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=ALL_REWARDS,
        train_dataset=dataset,
        args=config,
    )

    print(f"\n{'='*60}")
    print(f"  GRPO Training: {args.model}")
    print(f"  Steps: {args.max_steps}  |  Rewards: {len(ALL_REWARDS)} independent signals")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(args.output_dir)

    print(f"\n[DONE] Training complete! Model saved to: {args.output_dir}")
    if tokenizer:
        tokenizer.save_pretrained(args.output_dir)
        print(f"[DONE] Tokenizer saved to: {args.output_dir}")

    # ── Push to HuggingFace Hub ───────────────────────────────────────────────
    if args.push_to_hub:
        print(f"\n[HUB] Pushing model to HuggingFace Hub: {args.hub_repo} ...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=args.hub_repo,
                repo_type="model",
                commit_message="GRPO-trained Oversight Inbox Arena model",
            )
            print(f"[HUB] ✅ Model uploaded! View at: https://huggingface.co/{args.hub_repo}")
        except Exception as e:
            print(f"[HUB] ⚠️ Push failed: {e}")
            print(f"[HUB] You can push manually with:")
            print(f"        huggingface-cli upload {args.hub_repo} {args.output_dir} --repo-type model")


if __name__ == "__main__":
    main()
