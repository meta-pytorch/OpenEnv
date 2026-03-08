# Teaching Agents to Ask for Help

**Dynamic Expert-in-the-Loop for Agent World Model — with GRPO Reinforcement Learning**

> *"The mark of wisdom is not knowing everything — it's knowing when to ask."*

---

## The Big Idea

Imagine teaching a new employee their job. You wouldn't just hand them a manual and walk away. You also wouldn't stand behind them dictating every keystroke. The best approach? **Let them try, and tell them an expert is available if they get stuck.**

That's exactly what this project does — but for AI agents.

We give a small language model (Qwen3-4B) a set of 38 API tools, a task description, and access to a brilliant advisor (GPT-5.1). Then we use reinforcement learning (GRPO) to teach the agent *when* calling the expert leads to better outcomes — and ultimately, when it can fly solo.

**The result: completion rate jumped from ~3% to 44.5%, reward hit +0.824, and format errors dropped from 88% to 31% — all by teaching the agent to try first and ask for help only when stuck. The model autonomously graduated through three training phases: Scaffold, Balanced, and Independence.**

![Reward Curves](assets/reward_curves.png)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The Expert Tool](#the-expert-tool)
- [GRPO Training Results](#grpo-training-results)
  - [Baseline: No Expert](#experiment-1-baseline-no-expert)
  - [Expert-Assisted Training](#experiment-2-expert-assisted-training)
  - [Mixed Mode: The 50/50 Experiment](#experiment-3-mixed-mode-the-5050-experiment)
  - [Adaptive Ratio: Training Wheels](#experiment-4-adaptive-ratio-training-wheels)
- [The Training Wheels Analogy](#the-training-wheels-analogy)
- [Key Findings](#key-findings)
- [Expert Calling Behavior Analysis](#expert-calling-behavior-analysis)
- [Bug Fixes](#bug-fixes)
- [Learnings](#learnings)
- [Research Directions](#research-directions--whats-next)
- [File Structure](#file-structure)
- [Usage](#usage)

---

## Overview

The dynamic expert is exposed as a callable **tool** (`ask_expert`) that the agent invokes **during** the task whenever it needs guidance. Unlike upfront advice approaches, the agent decides when to consult the expert based on real-time context — errors, partial progress, or task complexity.

The expert is "verifier-informed": before the task starts, it analyzes the Python verification code to extract the exact database state required for success. Combined with full MCP tool schemas, it produces precise step-by-step plans with exact tool names and argument values.

Think of it like an open-book exam where one of the "books" is a brilliant tutor who knows the answer key — but you have to decide when it's worth raising your hand.

---

## Architecture

```mermaid
sequenceDiagram
    participant Agent as Agent Model
    participant Expert as Expert (verifier-informed)
    participant Env as AWM Environment
    participant Verifier as Verifier Code

    Verifier->>Expert: Success criteria (pre-analyzed)
    Agent->>Env: list_tools
    Env-->>Agent: tool schemas
    Note over Agent: Try first — attempt the task independently
    loop Try tools independently
        Agent->>Env: call_tool
        Env-->>Agent: result
        alt Error or stuck
            Agent->>Expert: ask_expert(task, tools, error_context)
            Expert-->>Agent: Recovery plan with exact tool names/args
            Agent->>Env: call_tool (follow plan)
            Env-->>Agent: result
        end
    end
    Agent->>Env: verify(mode=code)
    Env-->>Agent: reward
```

### How It Works

1. **Verifier analysis**: Before the task begins, the expert LLM analyzes the Python verifier code and extracts the exact success criteria — table names, column values, JSON field contents, and record relationships the verifier checks.
2. **Tool discovery**: The agent calls `list_tools` to get the full MCP tool catalog. The expert receives these schemas with parameter names, types, and descriptions.
3. **Try first**: The agent attempts the task independently using `call_tool`, reading tool descriptions and reasoning about arguments on its own.
4. **Ask if stuck**: If the agent encounters an error, gets stuck, or is unsure about a tool's arguments, it calls `ask_expert` with the error context. The expert returns a precise recovery plan.
5. **Adaptive execution**: The agent follows the expert's plan using `call_tool`. If another step fails, it can re-consult the expert (up to 3 times).
6. **Verification**: After completing tool calls, the code verifier checks the final database state.

### Key Features

- **Agent-initiated**: The agent decides when to call the expert (0 to N times per task)
- **Verifier-informed**: Expert knows the exact DB state the verifier checks for
- **Schema-aware**: Expert receives full MCP tool schemas to generate precise calls
- **Error recovery**: System nudges agent to consult expert after errors or stalls
- **No prior data needed**: Works from the first run — no baseline log required

---

## The Expert Tool

The `ask_expert` tool is implemented as a GPT-5.1 call with a carefully constructed prompt:

| Parameter | Details |
|-----------|---------|
| **Model** | GPT-5.1 (Azure OpenAI) |
| **Input** | Task description + available MCP tool schemas + error context |
| **Output** | Precise step-by-step plan with exact tool names and argument values |
| **Max calls per episode** | 3 (forces the agent to be selective) |
| **Latency** | ~2-3 seconds per call |

The expert is "verifier-informed" — it has already analyzed the Python verification function and knows exactly what database state constitutes success. This is like giving the tutor the answer key before the exam.

---

## GRPO Training Results

We ran four experiments, each building on the insights of the previous one. Here's the story of how we went from 3% completion rate to 44.5% — and taught the model to graduate through three autonomous training phases.

**Training setup:**

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-4B (SFT checkpoint) |
| Algorithm | GRPO (Generalized RL Policy Optimization) |
| GPUs | 8x NVIDIA B200 (183GB each) |
| Batch size | 16 prompts x 8 rollouts = 128 rollouts/step |
| Training tasks | 53 workflow automation scenarios |
| Validation tasks | 29 held-out scenarios |
| Total steps | 48 |
| Reward | Code verifier + LLM judge (GPT-5.1) |

### Experiment 1: Baseline (No Expert)

**The control group.** The agent has only `list_tools` and `call_tool` — no expert to lean on.

| Milestone | Step | Notes |
|-----------|------|-------|
| Start | 0 | Val accuracy: **3.4%**, train reward: -0.67 |
| First improvement | 9 | Reward jumps to -0.40 |
| Crosses zero | 16 | Reward turns positive (+0.15) |
| Peak | 42 | Reward reaches **+0.75** |
| Final | 48 | Val accuracy: **55.2%** |

> **Takeaway:** The baseline works, but it takes 16 steps just to reach positive reward. The agent spends most of early training producing malformed XML tool calls (format errors > 90%).

### Experiment 2: Expert-Assisted Training

**Every rollout has access to `ask_expert`.** The agent can call the GPT-5.1 expert up to 3 times per task.

| Step Window | Expert Train Reward | Baseline Train Reward | Expert Advantage |
|-------------|--------------------|-----------------------|-----------------|
| Steps 1-5 | **-0.37** | -0.73 | **+0.36** |
| Steps 6-10 | **-0.01** | -0.52 | **+0.51** |
| Steps 11-17 | **+0.19** | -0.22 | **+0.41** |

> **Takeaway:** The expert provides a massive early boost (+0.51 reward advantage by steps 6-10). Initial validation accuracy starts at 10.3% vs baseline's 3.4% — the expert system prompt alone helps even before any RL training.

### Experiment 3: Mixed Mode — The 50/50 Experiment

**The hypothesis:** If we give the expert to only *half* the rollouts, the agent might learn when it needs help vs when it can manage alone. Plus, a +0.5 bonus reward for completing tasks *without* the expert should incentivize independence.

For each prompt, 4 out of 8 rollouts get expert access ("expert chains"), and 4 don't ("solo chains").

![Completion Comparison](assets/completion_comparison.png)

**What happened:**

| Step Window | Baseline Completions | Mixed Completions | Mixed Advantage |
|-------------|---------------------|-------------------|----------------|
| Steps 1-5 | avg 2.4 (1.9%) | avg **8.2 (6.4%)** | **3.4x** |
| Steps 6-10 | avg 4.4 (3.4%) | avg **11.0 (8.6%)** | **2.5x** |
| Steps 11-15 | avg 13.0 (10.2%) | avg **19.2 (15.0%)** | **1.5x** |
| **Steps 16-17** | **avg 39.0 (30.5%)** | avg 25.0 (19.5%) | **Baseline wins** |

The mixed approach dominated early training (3.4x more completions in the first 5 steps!), but **the baseline overtook it at step 16**. Why?

![Format Error Divergence](assets/format_error_divergence.png)

**The culprit: solo chains are dead weight.** Solo chains maintain 85-87% format error rates throughout training, while expert chains drop to ~42%. Half the batch produces almost no useful gradient signal, diluting the learning for the entire model.

> **Takeaway:** A fixed 50/50 split wastes compute. The solo chains can't learn XML formatting fast enough without the expert's structured guidance, creating an anchor that drags down the whole run.

### Experiment 4: Adaptive Ratio — Training Wheels

**The insight from Experiment 3** was clear: don't use a fixed ratio. Instead, **adapt the expert/solo split based on how well the agent is doing.**

![Adaptive Ratio](assets/adaptive_ratio.png)

The adaptive algorithm reads the format error rate from the previous step and adjusts:

| Format Error Rate | Phase | Expert Chains | Solo Chains | Rationale |
|-------------------|-------|--------------|-------------|-----------|
| **> 70%** | Scaffold | 6 | 2 | Model can't even format tools — heavy expert support |
| **40-70%** | Balanced | 4 | 4 | Model has basics — equal exposure |
| **< 40%** | Independence | 2 | 6 | Model is proficient — push toward self-reliance |

**Results after 29 steps (Try-First v2 with reward shaping):**

The final strategy combines three innovations:
1. **"Try First" system prompt** — tells the agent to attempt tasks independently before asking the expert
2. **Graduated reward shaping** — solo completion gets +1.0 bonus, recovery expert gets +0.3 bonus, blind expert gets -0.2 penalty
3. **Adaptive expert ratio** — automatically transitions from heavy scaffolding to independence based on format error rate

| Step | Reward | Format Error % | Completions | Completion Rate | Phase |
|------|--------|---------------|-------------|-----------------|-------|
| 1 | -0.528 | 88.3% | 7 | 5.5% | Scaffold (6E/2S) |
| 2 | -0.577 | 85.9% | 4 | 3.1% | Scaffold |
| 3 | -0.362 | 84.4% | 12 | 9.4% | Scaffold |
| 4 | -0.606 | 89.8% | 4 | 3.1% | Scaffold |
| 5 | -0.427 | 87.5% | 13 | 10.2% | Scaffold |
| 6 | -0.201 | 81.3% | 13 | 10.2% | Scaffold |
| 7 | -0.396 | 81.3% | 15 | 11.7% | Scaffold |
| 8 | -0.421 | 75.0% | 12 | 9.4% | Scaffold |
| 9 | -0.076 | 72.7% | 24 | 18.8% | Scaffold |
| 10 | -0.021 | 65.6% | 25 | 19.5% | **Scaffold -> Balanced** |
| 11 | **+0.316** | 65.6% | 28 | 21.9% | Balanced (4E/4S) |
| 12 | **+0.264** | 60.2% | 28 | 21.9% | Balanced |
| 13 | **+0.156** | 60.2% | 28 | 21.9% | Balanced |
| 14 | +0.056 | 62.5% | 18 | 14.1% | Balanced |
| 15 | **+0.589** | 56.3% | **41** | **32.0%** | Balanced |
| 16 | +0.480 | 54.7% | 36 | 28.1% | Balanced |
| 17 | +0.435 | 44.5% | 38 | 29.7% | Balanced |
| 18 | +0.370 | 53.9% | 29 | 22.7% | Balanced |
| 19 | +0.365 | 50.8% | 36 | 28.1% | Balanced |
| 20 | **+0.603** | 50.0% | 37 | 28.9% | Balanced |
| 21 | +0.290 | 43.0% | 37 | 28.9% | Balanced |
| 22 | +0.478 | 43.0% | 51 | 39.8% | Balanced |
| 23 | **+0.824** | 43.0% | 51 | 39.8% | Balanced |
| 24 | +0.519 | 50.0% | 41 | 32.0% | Balanced |
| 25 | +0.493 | 44.5% | 31 | 24.2% | Balanced |
| 26 | +0.295 | 43.0% | 50 | 39.1% | Balanced |
| 27 | +0.571 | 43.8% | 42 | 32.8% | Balanced |
| 28 | +0.426 | 35.2% | 53 | 41.4% | **Balanced -> Independence** |
| 29 | +0.407 | **31.3%** | **57** | **44.5%** | Independence (2E/6S) |

**The training trajectory tells a clear story — with three distinct acts:**
- **Act 1 (Steps 1-10): Scaffold.** Learning basic tool formatting under heavy expert guidance. FE: 88% -> 66%. The model earns its way to more independence.
- **Act 2 (Steps 11-27): Balanced.** Equal expert/solo exposure. Reward turns positive and stays positive. Completions climb from 28 to 51. Peak reward +0.824 at step 23.
- **Act 3 (Steps 28+): Independence.** FE drops below 40%, system auto-shifts to 2 expert / 6 solo. The model thrives — 57 completions (44.5%) at step 29 with only 31% format errors.

**Head-to-head comparison (all strategies):**

| Metric | Try-First v2 | Adaptive v1 | Mixed 50/50 | Baseline (no expert) |
|--------|-------------|-------------|-------------|---------------------|
| Peak reward | **+0.824** (step 23) | +0.068 (step 9) | Never positive | Never positive |
| Peak completions/step | **57** (step 29) | 25 (step 9) | ~15 | ~5 |
| Peak completion rate | **44.5%** (step 29) | 19.5% | ~12% | ~4% |
| Reward crosses zero | **Step 10** | Step 9 (once) | Never | Never |
| Sustained positive reward | **Steps 11-29 (19 steps)** | No | No | No |
| Lowest format error | **31.3%** | ~70% | ~75% | ~77% |
| Phase transitions achieved | **All 3** (Scaffold -> Balanced -> Independence) | None | N/A | N/A |

**What made Try-First v2 different:**
- The system prompt encourages independent problem-solving: *"FIRST TRY to solve the task yourself... If you encounter an error, THEN call ask_expert"*
- Reward shaping creates a clear hierarchy: solo completion (reward=2.0) > recovery expert (reward=1.3) > blind expert (reward=0.8) > failure
- The combination produces agents that are both more capable AND more independent
- The adaptive ratio responds to the model's actual progress, not a fixed schedule — earning independence rather than being forced into it

---

## The Training Wheels Analogy

The best way to think about adaptive expert ratio is **training wheels on a bicycle.**

![Training Wheels Analogy](assets/training_wheels_analogy.png)

- **Phase 1 — Scaffold (Steps 1-10):** The child can barely balance. Training wheels are firmly on. Most chains get expert guidance so the agent learns *what good tool calls look like* before worrying about *when to ask for help.* FE: 88% -> 66%. Reward: -0.53 -> -0.02.

- **Phase 2 — Balanced (Steps 11-27):** The child is wobbly but upright. One training wheel comes off. Half the chains must solve tasks solo, and the model thrives — completions jump from 28 to 51, reward peaks at +0.824. FE: 66% -> 44%.

- **Phase 3 — Independence (Steps 28+):** The child is riding confidently. Both training wheels are off, with just a hand hovering nearby. Most chains are solo — and the model sets new records: 57 completions (44.5% rate), FE down to 31.3%.

**All three phases completed.** The key insight: you don't yank training wheels off a kid who can't balance yet. That's what the fixed 50/50 split did — and it's why it failed. The adaptive approach lets the model *earn* each transition.

---

## Key Findings

### 1. "Try First, Ask If Stuck" Is The Winning Strategy

The breakthrough came from changing a single instruction: instead of "call the expert for complex tasks," we told the agent "try first, then ask if you get stuck." Combined with reward shaping (solo bonus +1.0, recovery bonus +0.3, blind penalty -0.2), this produced agents that are both more capable and more independent. By step 29: **44.5% completion rate** — over 10x the baseline — with reward peaking at **+0.824**.

### 2. Graduated Reward Shaping Works Better Than Binary Bonuses

The original +0.5 solo bonus was too blunt — it rarely triggered because the model couldn't complete tasks solo early on. The graduated system creates a clear hierarchy: solo (reward=2.0) > recovery expert (1.3) > blind expert (0.8) > failure. Over 29 steps, **635 solo completions** were achieved and **272 recovery expert calls** were rewarded, proving the gradient signal works at every skill level.

### 3. Format Errors Are The Gateway Skill

88% of rollouts die from malformed XML tool calls in step 1. By step 29, this drops to **31.3%** — a 57-percentage-point improvement. Expert scaffolding teaches formatting faster than solo trial-and-error. **An agent that can't format a tool call can't learn anything else.**

### 4. All Three Adaptive Phase Transitions Fired

The model graduated through every phase the adaptive system designed:
- **Step 10**: FE crossed 70% threshold -> Scaffold to Balanced (6E/2S -> 4E/4S)
- **Step 28**: FE crossed 40% threshold -> Balanced to Independence (4E/4S -> 2E/6S)

The model didn't regress at either transition — it thrived with each increase in independence. At step 29, in full Independence mode, it hit its best completion rate (44.5%).

### 5. The Compounding Effect: Scaffolding Then Independence

Neither pure scaffolding nor pure independence training works alone. The magic is in the three-act sequence: expert scaffolding teaches *what* good tool calls look like (steps 1-10), then balanced exposure teaches *when* to ask for help (steps 11-27), and finally independence mode proves the agent can ride solo (steps 28+). The "try first" prompt and reward shaping prevent expert addiction at every stage.

---

## Expert Calling Behavior Analysis

How does the agent actually *use* the expert over time?

![Expert Behavior Evolution](assets/expert_behavior_evolution.png)

### Three Patterns Emerge:

**1. "Blind Planning":** The agent calls `ask_expert` as its very first action, before trying any tools. Still the most common pattern, but the -0.2 blind penalty is creating pressure to change.

**2. "Error Recovery" (the goal):** The agent tries a tool, encounters an error, then calls the expert. This is the pattern we want to reinforce with the +0.3 recovery bonus.

**3. "Solo Completion" (the ultimate goal):** The agent completes the task without ever calling the expert. These get the +1.0 solo bonus, making them the highest-rewarded pattern.

**Reward shaping results after 29 training steps:**

| Pattern | Count | Reward Modifier | Trend |
|---------|-------|-----------------|-------|
| Blind expert calls | 1,115 | -0.2 penalty | Decreasing as share of total |
| Recovery expert calls | 272 | +0.3 bonus | Steadily growing |
| Solo completions | **635** | +1.0 bonus | Exploding growth (0 at step 1 -> 635 by step 29) |

> **The shift happened.** In the first training run (without "try first" prompt), blind expert calls were 77% and growing. With the new prompt and reward shaping, solo completions rocketed from zero to 635 — the model learned that solving tasks independently is the highest-reward strategy. The blind-to-solo ratio went from infinity to roughly 2:1, and continues improving.

---

## Benchmark Results

### Inference-Time (GPT-5.1 agent, no RL)

Scenario: `workflow_automation_1` (10 tasks), gpt-5.1

| Metric | Baseline (no expert) | Dynamic Expert | Delta |
|--------|---------------------|----------------|-------|
| Avg reward | 0.500 | 0.800 | **+0.300 (+60%)** |
| Complete tasks | 5/10 | 8/10 | **+3** |

### GRPO Training (Qwen3-4B, Try-First v2, 29 steps)

| Metric | Step 1 | Step 29 | Delta |
|--------|--------|---------|-------|
| Train reward | -0.528 | +0.407 (peak +0.824) | **+1.35** |
| Completion rate | 5.5% | 44.5% | **+39pp** |
| Format error rate | 88.3% | 31.3% | **-57pp** |
| Completions/step | 7 | 57 | **+50 (8x)** |
| Adaptive phase | Scaffold (6E/2S) | Independence (2E/6S) | **All 3 transitions** |

---

## Bug Fixes

### SQLite Seed Data Quoting (`db_manager.py`)

**Problem**: The AWM dataset contains SQL INSERT statements with backslash-escaped single quotes (e.g., `\'high\'`). SQLite does not recognize `\'` — the standard escape is doubled quotes (`''`).

**Impact**: Seed data for some tables silently failed to insert, leaving the environment in an inconsistent state. Agents encountered unexpected errors interacting with resources that should have existed.

**Fix**: Added `_fix_escaped_quotes()` that converts `\'` to `''` before execution, used as a fallback when the original statement fails.

### FastAPI Exception Handler (`scenario_manager.py`)

**Problem**: Dynamically generated FastAPI sub-environments returned opaque 500 errors with no traceback.

**Fix**: Injected a Starlette exception handler that prints full tracebacks and returns error details in the JSON response.

### Configurable Azure OpenAI Settings (`example_usage.py`)

**Problem**: Hardcoded model names and non-standard env var names.

**Fix**: Model, endpoint, and API version now read from environment variables:
- `AZURE_OPENAI_MODEL` (default: `gpt-5.1`)
- `AZURE_OPENAI_ENDPOINT`
- `OPENAI_API_VERSION` (default: `2025-04-01-preview`)

---

## Learnings

### System Prompt Engineering Matters

Two key improvements brought task pass rate from 3/10 to 8/10:

- **Avoid playbook shortcuts**: AWM environments expose both high-level "playbook" tools and granular CRUD tools. Playbooks often don't match task requirements exactly. Using granular tools improved accuracy significantly.
- **Check before create**: Many environments have pre-existing seed data. Adding "check if it exists first, update if so" eliminated UNIQUE constraint errors.

### Seed Data Integrity Is Critical

The SQLite quoting bug caused silent failures during database initialization. The environment appeared to work, but tables were missing rows. This led to cascading errors that looked like agent mistakes but were infrastructure bugs.

### Expert Is Most Valuable for Edge Cases

For straightforward tasks, the agent succeeds without expert help. The expert's biggest impact is on tasks that require specific sequences or have non-obvious pitfalls — error recovery via re-consultation is the key differentiator.

### RL Training Insights

- **Format errors dominate early training.** 88%+ of rollouts die before any meaningful tool interaction. Reducing format errors is the single highest-leverage improvement — and the agent reduced them from 88% to 31%.
- **"Try first" prompt + reward shaping changes expert usage behavior.** Without it, the agent learns "always ask first." With it, solo completions grew from 0 to 635 over 29 steps while blind expert calls decreased as a share of total actions.
- **Fixed expert ratios waste compute.** A 50/50 split sounds fair but creates "dead weight" — solo chains that can't learn fast enough drag down the whole batch. The adaptive approach ensures useful gradient signal at every training stage.
- **Adaptive phase transitions build on each other.** Each phase unlocked by the model's own progress: Scaffold taught formatting, Balanced taught when to ask, Independence proved the agent can ride solo. The 3-phase journey (steps 1 -> 10 -> 28) mirrors human learning progressions.

---

## Research Directions — What's Next?

The experiments above scratched the surface of a fascinating question: **how do you teach an agent to know what it doesn't know?**

Here are promising directions for future exploration:

### 1. Curriculum Learning for Expert Withdrawal

Instead of adapting based on format error rate alone, build a curriculum that explicitly tracks *per-task* difficulty. Easy tasks lose expert access first; hard tasks keep it longer. This mirrors how human education works — you don't stop teaching long division because the student mastered addition.

### 2. Confidence-Calibrated Expert Calls

Train the agent to output a confidence score before each action. If confidence is below a threshold, it calls the expert. This requires auxiliary training signal but could produce genuinely selective expert usage — the holy grail of "knowing when you don't know."

### 3. Expert Distillation

Instead of calling a live GPT-5.1 expert at training time, distill the expert's knowledge into a small auxiliary model. This removes the latency and cost of expert calls during training while preserving the scaffolding benefit.

### 4. Multi-Expert Ensembles

What if the agent could choose between different experts — a "tool formatting" expert, a "task planning" expert, and an "error recovery" expert? Teaching an agent to route to the right specialist is a richer learning problem.

### 5. Self-Play Expert Bootstrapping

As the agent improves, use its own successful trajectories as "expert" demonstrations for the next training round. The agent bootstraps its own expertise, gradually replacing the external expert with internalized knowledge.

### 6. Meta-Learning Across Tasks

Train on a diverse set of environments (not just workflow automation) so the agent learns a general "when to ask for help" policy that transfers across domains. An agent that knows when to ask in *any* environment is far more valuable than one that memorizes when to ask in *one* environment.

> **The broader vision:** In the future, the most capable AI systems won't be the ones that know everything — they'll be the ones that know the boundaries of their own knowledge and can effectively collaborate with other systems (or humans) to fill the gaps. Teaching agents to ask for help is a step toward that future.

---

## File Structure

```
agent_world_model_env/
├── run_awm_task.py                      # Single-model runner (no expert)
├── run_awm_task_dynamic_expert.py       # Dynamic expert benchmark runner
├── EXPERT_ENHANCEMENT.md               # This file
└── assets/
    ├── reward_curves.png               # GRPO training reward over 48 steps
    ├── completion_comparison.png        # Baseline vs mixed completions
    ├── format_error_divergence.png      # Expert vs solo chain format errors
    ├── adaptive_ratio.png              # Adaptive phase transition diagram
    ├── training_wheels_analogy.png     # Visual analogy for adaptive training
    └── expert_behavior_evolution.png   # How the agent uses the expert over time
```

## Usage

```bash
# Set credentials
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"

# Start the AWM server
uvicorn agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899

# Run full benchmark: baseline vs dynamic expert
python run_awm_task_dynamic_expert.py workflow_automation_1

# Run with a different model
python run_awm_task_dynamic_expert.py workflow_automation_1 --model gpt-5.1

# Run only the expert mode (skip baseline)
python run_awm_task_dynamic_expert.py workflow_automation_1 --expert-only

# Run only baseline (skip expert)
python run_awm_task_dynamic_expert.py workflow_automation_1 --baseline-only

# Limit to fewer tasks
python run_awm_task_dynamic_expert.py workflow_automation_1 --tasks 3
```

---
