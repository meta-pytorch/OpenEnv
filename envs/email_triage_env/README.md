# Email Triage Environment

## Problem and Motivation

This environment models a real customer operations workflow: triaging inbound emails at scale.
Agents must decide category, urgency, and escalation behavior under noisy message content.

## Action and Observation Space

Action (`EmailTriageAction`):

- `category`: one of `billing`, `support`, `spam`, `urgent`, `marketing`, `other`
- `priority`: integer in `[1, 5]`
- `should_escalate`: boolean

Observation (`EmailTriageObservation`):

- `email_id`, `subject`, `body_snippet`, `sender`, `sender_domain`, `is_internal`
- `task_id`: one of `easy`, `medium`, `hard`
- `reward`, `done`, `metadata`, `info`

State (`EmailTriageState`):

- `episode_id`, `step_count`, `total_reward`, `difficulty`, `current_task`

## Tasks and Difficulty Progression

The environment exposes three deterministic tasks via `reset(difficulty=...)` or `reset(task_id=...)`:

1. `easy`: category classification
2. `medium`: category + priority quality
3. `hard`: full triage (category + priority + escalation safety)

## Programmatic Graders

Base graders (`server/graders.py`):

- `category_grader`: exact category match
- `priority_grader`: bucket-based partial credit (`low/med/high`)
- `escalation_grader`: escalation correctness with harmful mismatch handling

Task graders (`server/graders.py`):

- `easy_task_grader`
- `medium_task_grader`
- `hard_task_grader`
- `task_grader` dispatcher

All task graders are deterministic and return scores in `[0.0, 1.0]`.

## Reward Design

Reward is shaped per task with weighted partial progress:

- Easy emphasizes category correctness
- Medium emphasizes category + priority
- Hard emphasizes category + priority + escalation

Additional penalties are applied for harmful behavior:

- escalating spam
- not escalating urgent incidents

## Dataset

- Path: `envs/email_triage_env/server/email_triage_dataset.json`
- Size: 120 synthetic-but-realistic labeled emails
- Labels include: `true_category`, `true_priority`, `needs_escalation`, `difficulty`

## OpenEnv Metadata

Environment metadata is defined in `openenv.yaml`:

- runtime: `fastapi`
- app: `server.app:app`
- port: `8000`

## Build and Run

From OpenEnv repo root:

```bash
docker build -t email-triage-env-openenv -f envs/email_triage_env/server/Dockerfile .
docker run -p 8000:8000 email-triage-env-openenv
```

Docs endpoint:

```text
http://localhost:8000/docs
```

## Baseline Inference Script

Required script: `inference.py` (included at env root).

Expected env variables:

- `API_BASE_URL`
- `MODEL_NAME`
- one API key variable: `API_KEY` (recommended) or `HF_TOKEN` or `GROQ_API_KEY` or `OPENAI_API_KEY`
- optional: `LOCAL_IMAGE_NAME`

Groq example:

```bash
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_key
```

Template file: `.env.example`

Run baseline:

```bash
cd envs/email_triage_env
python inference.py
```

The script prints structured logs in required format:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Quick Async Client Test

```python
import asyncio

from envs.email_triage_env import EmailTriageAction, EmailTriageEnv


async def main() -> None:
    env = await EmailTriageEnv.from_docker_image("email-triage-env-openenv:latest", port=8010)
    result = await env.reset(difficulty="hard", seed=123)
    print(result.observation.subject)

    action = EmailTriageAction(category="billing", priority=3, should_escalate=False)
    result = await env.step(action)
    print(result.observation.reward)
    print(result.observation.info)
    await env.close()


asyncio.run(main())
```
