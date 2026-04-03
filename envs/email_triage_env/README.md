# Email Triage Environment

## Overview

This environment simulates high-volume inbox triage for support operations. The agent receives one incoming email per episode and must make operational decisions that mirror real customer support workflows at scale.

## Task Definition

For each email, the agent predicts:

- `category`: one of `billing`, `support`, `spam`, `urgent`, `marketing`, `other`
- `priority`: integer in range `1` to `5`
- `should_escalate`: boolean flag for whether the email should be escalated

## Dataset

The environment uses a synthetic-but-realistic dataset of 100+ labeled emails stored at:

- `envs/email_triage_env/server/email_triage_dataset.json`

Each record includes sender and message content plus ground-truth labels:

- `true_category`
- `true_priority`
- `needs_escalation`
- `difficulty`

The dataset was generated offline using LLM tooling and then validated into a stable JSON benchmark.

## Programmatic Graders

Reward is computed from explicit grader functions:

- `category_grader`: exact-match category accuracy
- `priority_grader`: bucketed scoring (low/med/high) with partial credit
- `escalation_grader`: escalation decision correctness with strong penalties for harmful mismatches

## Reward Function

Final reward is a linear combination:

- `1.0 * category_score`
- `0.5 * priority_score`
- `0.5 * escalation_score`

Additional safety penalties are applied for harmful decisions:

- escalating spam
- failing to escalate urgent incidents

## Edge Cases Covered

- Spam misclassification as urgent: receives strong negative shaping due to wrong category and harmful escalation.
- Urgent incident with no escalation: receives an explicit safety penalty even if category or priority are otherwise reasonable.
- Ambiguous subject lines: records with weak lexical cues still require balanced category, priority, and escalation decisions.

## Folder Layout

```text
envs/
  email_triage_env/
    __init__.py
    models.py
    client.py
    README.md
    server/
      email_triage_environment.py
      graders.py
      app.py
      Dockerfile
      email_triage_dataset.json
```

## Build and Run

From repo root:

```bash
docker build -t email-triage-env -f envs/email_triage_env/server/Dockerfile .
docker run -p 8000:8000 email-triage-env
```

If Docker says port 8000 is already in use, run on a different host port:

```bash
docker run -p 8001:8000 email-triage-env
```

## Python Quick Test

```python
import asyncio

from envs.email_triage_env import EmailTriageEnv, EmailTriageAction


async def main() -> None:
  env = await EmailTriageEnv.from_docker_image("email-triage-env:latest")
  result = await env.reset()
  print(result.observation.subject)

  action = EmailTriageAction(category="billing", priority=3, should_escalate=False)
  result = await env.step(action)
  print(result.observation.reward, result.observation.info)
  await env.close()


asyncio.run(main())
```
