#!/usr/bin/env python3
"""GRPO training script for Oversight Inbox Arena.

Addresses official hackathon guide requirements:
- Multiple independent reward functions (not just one combined score)
- Curriculum learning (easy -> medium -> hard progression)
- Anti-reward-hacking monitoring (output inspection)
- Unsloth support for low-VRAM training

Usage:
    python train_grpo.py                              # Full training
    python train_grpo.py --smoke                      # Quick smoke test
    python train_grpo.py --unsloth                    # Low-VRAM Unsloth mode
    python train_grpo.py --curriculum                 # Curriculum: easy->medium->hard
    python train_grpo.py --model Qwen/Qwen3-4B       # Larger model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
# TRL Environment Factory
# ---------------------------------------------------------------------------

class OversightInboxEnv:
    """TRL-compatible environment wrapper.

    TRL discovers tool methods via docstrings and calls them during rollouts.
    """

    def __init__(self, difficulty: str = "hard") -> None:
        self._difficulty = difficulty
        self._env = EmailTriageEnvironment(difficulty=difficulty)
        self._current_obs: Optional[Any] = None
        self._done: bool = False
        self._step_count: int = 0

        # Per-episode tracking for independent reward functions
        self._total_reward: float = 0.0
        self._quality_sum: float = 0.0
        self._sla_sum: float = 0.0
        self._policy_sum: float = 0.0
        self._oversight_sum: float = 0.0
        self._efficiency_sum: float = 0.0

    def reset(self, scenario_id: str = "default", seed: int = 0) -> str:
        """Reset the inbox queue with a new scenario.

        Args:
            scenario_id: Scenario identifier for logging.
            seed: Random seed for deterministic replay.

        Returns:
            Initial briefing with queue summary and specialist reports.
        """
        seed_int = int(seed) if seed else 0
        self._current_obs = self._env.reset(
            seed=seed_int, difficulty=self._difficulty
        )
        self._total_reward = 0.0
        self._quality_sum = 0.0
        self._sla_sum = 0.0
        self._policy_sum = 0.0
        self._oversight_sum = 0.0
        self._efficiency_sum = 0.0
        self._done = False
        self._step_count = 0

        obs = self._current_obs
        info = obs.info or {}
        specialist = info.get("specialist_reports", {})
        triage = specialist.get("triage", {})
        compliance = specialist.get("compliance", {})

        brief = (
            f"Queue loaded ({info.get('queue_size', '?')} tickets). "
            f"Current ticket: {obs.email_id}\n"
            f"Subject: {obs.subject}\n"
            f"Body: {obs.body_snippet[:200]}\n"
            f"Sender: {obs.sender} ({obs.sender_domain}) "
            f"{'[INTERNAL]' if obs.is_internal else '[EXTERNAL]'}\n"
            f"Specialist triage suggests: category={triage.get('category', '?')} "
            f"priority={triage.get('priority', '?')} "
            f"(confidence={triage.get('confidence', '?')})\n"
            f"Compliance flag: {compliance.get('flagged', False)} "
            f"reason={compliance.get('reason', 'none')}"
        )
        return brief

    def triage_ticket(
        self,
        category: str,
        priority: int,
        should_escalate: bool,
        rationale: str = "",
    ) -> str:
        """Submit a triage decision for the current ticket.

        Args:
            category: One of billing, support, spam, urgent, marketing, other.
            priority: Integer 1-5 (1=lowest, 5=critical).
            should_escalate: Whether to escalate to human reviewer.
            rationale: Brief reasoning for this decision.

        Returns:
            Feedback with reward, scores, and next ticket info.
        """
        if self._done:
            return "Episode already finished. Call reset() first."

        pri = max(1, min(5, int(priority)))
        valid_cats = {"billing", "support", "spam", "urgent", "marketing", "other"}
        cat = category.lower().strip() if category else "other"
        if cat not in valid_cats:
            cat = "other"

        action = EmailTriageAction(
            category=cat,
            priority=pri,
            should_escalate=bool(should_escalate),
            rationale=rationale or None,
        )

        obs = self._env.step(action)
        self._current_obs = obs
        self._total_reward += float(obs.reward)
        self._done = obs.done
        self._step_count += 1

        # Track component scores for independent reward functions
        info = obs.info or {}
        components = info.get("reward_components", {})
        self._quality_sum += components.get("quality", 0.0)
        self._sla_sum += components.get("sla", 0.0)
        self._policy_sum += components.get("policy", 0.0)
        self._oversight_sum += components.get("oversight", 0.0)
        self._efficiency_sum += components.get("efficiency", 0.0)

        result = (
            f"Step {self._step_count} result:\n"
            f"Reward: {obs.reward:.3f} (total: {self._total_reward:.3f})\n"
            f"Scores -- quality:{components.get('quality', 'N/A')} "
            f"sla:{components.get('sla', 'N/A')} "
            f"policy:{components.get('policy', 'N/A')} "
            f"oversight:{components.get('oversight', 'N/A')}\n"
            f"True answer: cat={info.get('true_category', '?')} "
            f"pri={info.get('true_priority', '?')} "
            f"esc={info.get('true_needs_escalation', '?')}\n"
        )

        if info.get("policy_drift_occurred"):
            result += f"WARNING POLICY DRIFT: {info.get('drift_description', '')}\n"

        if not obs.done:
            specialist = info.get("specialist_reports", {})
            triage_r = specialist.get("triage", {})
            result += (
                f"\nNext ticket: {obs.email_id}\n"
                f"Subject: {obs.subject}\n"
                f"Body: {obs.body_snippet[:200]}\n"
                f"Specialist suggests: category={triage_r.get('category', '?')} "
                f"priority={triage_r.get('priority', '?')}\n"
                f"Remaining: {info.get('tickets_remaining', '?')} tickets"
            )
        else:
            s = self._env.state
            result += (
                f"\nEPISODE COMPLETE\n"
                f"Tickets resolved: {s.tickets_resolved}/{s.queue_size}\n"
                f"SLA breaches: {s.sla_breaches}\n"
                f"Policy violations: {s.policy_violations}\n"
                f"Oversight catches: {s.oversight_catches}\n"
                f"Drift events: {s.drift_count}\n"
                f"Total reward: {s.total_reward:.3f}"
            )

        return result


# ---------------------------------------------------------------------------
# Multiple Independent Reward Functions (per official guide FAQ #7, #8)
#
# "Use multiple independent reward functions, not just one. If you only
#  have a single reward signal, it is easier for the model to hack it."
# ---------------------------------------------------------------------------

def reward_quality(completions: list, environments: Optional[list] = None, **kw) -> list:
    """Reward 1: Resolution quality — category + priority + escalation accuracy."""
    if not environments:
        return [0.0] * len(completions)
    return [
        e._quality_sum / max(1, e._step_count) if hasattr(e, "_quality_sum") else 0.0
        for e in environments
    ]


def reward_oversight(completions: list, environments: Optional[list] = None, **kw) -> list:
    """Reward 2: Specialist error correction — did coordinator catch mistakes?"""
    if not environments:
        return [0.0] * len(completions)
    return [
        e._oversight_sum / max(1, e._step_count) if hasattr(e, "_oversight_sum") else 0.0
        for e in environments
    ]


def reward_compliance(completions: list, environments: Optional[list] = None, **kw) -> list:
    """Reward 3: Policy compliance — did actions follow active policy rules?"""
    if not environments:
        return [0.0] * len(completions)
    return [
        e._policy_sum / max(1, e._step_count) if hasattr(e, "_policy_sum") else 0.0
        for e in environments
    ]


def reward_sla(completions: list, environments: Optional[list] = None, **kw) -> list:
    """Reward 4: SLA adherence — were tickets resolved before deadlines?"""
    if not environments:
        return [0.0] * len(completions)
    return [
        e._sla_sum / max(1, e._step_count) if hasattr(e, "_sla_sum") else 0.0
        for e in environments
    ]


def reward_no_hacking(completions: list, environments: Optional[list] = None, **kw) -> list:
    """Reward 5: Anti-cheat — penalizes repeated identical actions and timeouts."""
    if not environments:
        return [0.0] * len(completions)
    results = []
    for e in environments:
        penalty = 0.0
        if hasattr(e, "_env"):
            env = e._env
            # Penalize repetition
            if hasattr(env, "_repetition_penalties"):
                penalty -= 0.3 * env._repetition_penalties
            # Penalize timeout
            if env._state.step_count > env._max_episode_steps:
                penalty -= 1.0
        results.append(max(-2.0, penalty))
    return results


# All reward functions as a list — TRL uses each independently
ALL_REWARD_FUNCTIONS = [
    reward_quality,
    reward_oversight,
    reward_compliance,
    reward_sla,
    reward_no_hacking,
]


# ---------------------------------------------------------------------------
# Curriculum Learning (per official guide FAQ #14)
#
# "Start with the easiest version, then progress.
#  Make success possible early. If the model never sees successful
#  trajectories, learning stalls."
# ---------------------------------------------------------------------------

def build_curriculum_datasets(
    dataset_cls, system_msg: str, sizes: Dict[str, int]
) -> List:
    """Build datasets for each difficulty tier in curriculum order."""
    datasets = []
    for difficulty, size in sizes.items():
        prompts = []
        for i in range(size):
            prompts.append([
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        f"Process the {difficulty} inbox queue. "
                        f"Scenario seed: {i}. "
                        f"Difficulty: {difficulty}."
                    ),
                },
            ])
        ds = dataset_cls.from_dict({"prompt": prompts})
        datasets.append((difficulty, ds))
    return datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO training for Oversight Inbox Arena"
    )
    parser.add_argument(
        "--unsloth", action="store_true", help="Use Unsloth for low-VRAM training"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Quick smoke run (4 samples, 2 steps)"
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning: easy -> medium -> hard",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-1.7B", help="Base model name"
    )
    parser.add_argument(
        "--output-dir", default="oversight-inbox-grpo", help="Output directory"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=128, help="Training dataset size per tier"
    )
    parser.add_argument(
        "--num-generations", type=int, default=4, help="GRPO generations per prompt"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100, help="Max training steps per phase"
    )
    parser.add_argument(
        "--report-to", default="none", help="Reporting backend (wandb/none)"
    )
    args = parser.parse_args()

    # Smoke mode overrides
    if args.smoke:
        args.dataset_size = 4
        args.num_generations = 2
        args.max_steps = 2
        args.curriculum = False
        print("[SMOKE] Minimal run to verify pipeline")

    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("ERROR: Install trl and datasets first:")
        print("  pip install trl datasets transformers accelerate")
        sys.exit(1)

    if args.unsloth:
        try:
            from unsloth import FastLanguageModel, PatchFastRL

            PatchFastRL("unsloth", FastLanguageModel)
            print("[OK] Unsloth patches applied")
        except ImportError:
            print("[WARN] Unsloth not found, continuing without it")

    # System prompt
    system_msg = (
        "You are an expert email triage coordinator managing a team of specialist agents. "
        "For each ticket in your queue, use the triage_ticket tool to assign category "
        "(billing/support/spam/urgent/marketing/other), priority (1-5), and escalation "
        "decision. Consider specialist recommendations but override when they seem wrong. "
        "Minimize SLA breaches, never escalate spam, always escalate urgent incidents. "
        "Adapt immediately when policies change mid-episode."
    )

    print(f"[CONFIG] Model: {args.model}")
    print(f"[CONFIG] Output: {args.output_dir}")
    print(
        f"[CONFIG] Reward functions: {len(ALL_REWARD_FUNCTIONS)} independent signals"
    )
    print(f"[CONFIG] Curriculum: {args.curriculum}")

    # Build trainer config
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
        bf16=True,
    )

    # Add vLLM if available
    if not args.smoke:
        try:
            import vllm  # noqa: F401

            config.use_vllm = True
            config.vllm_mode = "colocate"
            config.vllm_gpu_memory_utilization = 0.3
            print("[OK] vLLM detected, using colocate mode")
        except ImportError:
            print("[INFO] vLLM not found, using standard generation")

    # ── Curriculum training ──────────────────────────────────────────
    if args.curriculum:
        print("\n--- CURRICULUM LEARNING ---")
        curriculum_sizes = {
            "medium": args.dataset_size,
            "hard": args.dataset_size,
        }
        if not args.smoke:
            curriculum_sizes = {
                "easy": args.dataset_size // 4,
                "medium": args.dataset_size // 2,
                "hard": args.dataset_size,
            }

        phases = build_curriculum_datasets(Dataset, system_msg, curriculum_sizes)

        for phase_idx, (difficulty, dataset) in enumerate(phases):
            phase_dir = f"{args.output_dir}/phase_{phase_idx}_{difficulty}"
            config.output_dir = phase_dir
            config.max_steps = args.max_steps

            # Factory that creates env at the right difficulty
            def make_env(diff=difficulty):
                return OversightInboxEnv(difficulty=diff)

            print(f"\n[PHASE {phase_idx + 1}/{len(phases)}] "
                  f"Difficulty: {difficulty} | "
                  f"Dataset: {len(dataset)} prompts | "
                  f"Steps: {config.max_steps}")

            model_name = (
                args.model
                if phase_idx == 0
                else f"{args.output_dir}/phase_{phase_idx - 1}_{phases[phase_idx - 1][0]}"
            )

            trainer = GRPOTrainer(
                model=model_name,
                reward_funcs=ALL_REWARD_FUNCTIONS,
                train_dataset=dataset,
                args=config,
                environment_factory=make_env,
            )

            trainer.train()
            trainer.save_model(phase_dir)
            print(f"[SAVED] Phase {phase_idx + 1} model -> {phase_dir}")

        print(f"\n[DONE] Curriculum training complete!")
        return

    # ── Standard (single-phase) training ─────────────────────────────
    prompts = []
    for i in range(args.dataset_size):
        prompts.append([
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"Process the inbox queue. Scenario seed: {i}.",
            },
        ])

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"[DATA] {len(dataset)} prompts")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=ALL_REWARD_FUNCTIONS,
        train_dataset=dataset,
        args=config,
        environment_factory=OversightInboxEnv,
    )

    print("[START] GRPO training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"[SAVED] Model -> {args.output_dir}")
    print("[DONE] Training complete!")


if __name__ == "__main__":
    main()
