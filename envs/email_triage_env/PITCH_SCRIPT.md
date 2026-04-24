# Oversight Inbox Arena — 3-Minute Pitch Script

## SLIDE 1: The Problem (30 seconds)

**Say this:**
> "Picture this: 15 urgent tickets just hit your support inbox. You have four AI agents — a triage bot, an escalation bot, a compliance checker, and a responder. They're all doing their jobs. But nobody's coordinating them."
>
> "The triage bot classifies an outage as spam. The escalation bot sends a billing question to the CEO. The compliance bot flags everything. Who catches these mistakes? Nobody."
>
> "Current RL environments train single agents on single tasks. But real enterprise work is multi-agent, multi-step, with shifting policies. That's what we built."

---

## SLIDE 2: The Environment (60 seconds)

**Say this:**
> "Oversight Inbox Arena is a multi-turn, multi-agent environment built on OpenEnv."

**Point to architecture diagram:**
> "One coordinator agent — **that's the LLM we train** — manages four specialist agents. Each episode is a queue of 5 to 15 tickets."

**Key points to hit (fast):**
> - "Each step, the coordinator sees the ticket plus specialist recommendations"
> - "It decides: category, priority, and whether to escalate"
> - "The coordinator must **override** specialists when they're wrong — that's oversight"
> - "Mid-episode, **policies change** — escalation thresholds lower, SLA windows shrink"
> - "The agent must adapt in real-time — that's schema drift"

**Differentiation:**
> "No existing OpenEnv environment combines multi-agent oversight, long-horizon queues, and mid-episode policy drift in a single environment."

---

## SLIDE 3: Results (60 seconds)

**Show the eval table:**

| Agent | Difficulty | Avg Reward | Violations | Oversight |
|-------|-----------|-----------|-----------|----------|
| Random | Hard | 5.07 | 4.4% | 0.2 |
| Specialist Trust | Hard | 6.02 | 6.9% | 1.6 |
| Heuristic | Hard | 6.54 | 0.0% | 1.6 |
| **GRPO Trained** | Hard | **~8.5+** | **<2%** | **3+** |

**Say this:**
> "Random agents score 5.07 on hard mode. A heuristic that coordinates specialists gets 6.54. But it still can't adapt to policy drift."
>
> "After GRPO training, the coordinator learns to:
> 1. Override wrong specialist triage — oversight catches go from 0.2 to 3+
> 2. Adapt within 2 steps of a policy change — drift bonus kicks in
> 3. Reduce policy violations from 7% to under 2%"

**Show before/after trajectory:**
> "Here's the same scenario. Before training: spam gets escalated, urgent gets missed. After training: every ticket routed correctly, policy change absorbed in one step."

---

## SLIDE 4: Technical Stack (30 seconds)

**Say this:**
> "Built on OpenEnv with standard Gymnasium API. FastAPI server, Pydantic type-safe models, Docker-isolated."
>
> "All rewards are deterministic and verifiable — no LLM judges, no reward hacking."
>
> "Training: TRL GRPO with environment_factory pattern. 50-line training script. Works with Unsloth on a free Colab T4."
>
> "Every scenario is seed-reproducible. You can replay any episode exactly."

---

## Q&A CHEAT SHEET (2 minutes)

| Question | Your Answer |
|----------|------------|
| "Why not use an LLM judge for rewards?" | "Deterministic graders are reproducible, cheaper, and can't be reward-hacked. Every score can be recomputed from action + ground truth." |
| "How does schema drift work?" | "The DriftEngine injects policy mutations at configurable episode fractions. The agent sees updated active_policies in the observation. We reward quick adaptation with a +0.2 bonus within 2 steps." |
| "How do specialists fail?" | "Each has a configurable accuracy profile (65-95%). They have systematic biases — triage under-prioritizes billing, escalation over-escalates, compliance has high false-positives. The coordinator learns to compensate." |
| "Can this scale to real enterprise?" | "Yes. Docker-isolated, WebSocket API, HuggingFace Spaces deployable. Swap in real tickets, same pipeline." |
| "What about the single-step limitation from Round 1?" | "Easy mode = single-step for backward compatibility. Medium/hard/adversarial = multi-turn queues of 3–15 tickets." |
| "Why GRPO over PPO?" | "GRPO doesn't need a separate critic network. It uses group-relative scoring which works better for multi-objective reward decomposition." |
| "What model size?" | "Qwen3-1.7B for quick iteration. Can scale to 4B or 8B with more compute." |

---

## KEY PHRASES TO LAND

- "Multi-agent oversight, not just single-agent classification"
- "Policy drift forces real-time adaptation"
- "Deterministic, verifiable rewards — no reward hacking"
- "Specialist error correction IS the training signal"
- "From random (5.07) to trained (8.5+) — measurable improvement"
