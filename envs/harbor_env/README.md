---
title: Harbor
emoji: ⚓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# Harbor Environment

Run a coding agent on a [Harbor](https://github.com/laude-institute/harbor) task and get back the
exact token ids and per-token logprobs of every model call it made, plus the task's own reward.
That is what a trainer needs, and it cannot be reconstructed afterwards.

## Overview

Harbor decouples three things that usually come welded together:

| | |
|---|---|
| **task** | the instruction, the sandbox image, and the verifier that scores the result |
| **harness** | the coding agent: its tool surface and its loop |
| **sandbox** | where the agent runs: docker, e2b, modal, daytona and others |

This environment adds the OpenEnv surface on top: dataset discovery over the Task API, one
`run_rollout` MCP tool, and a capture proxy that sits between the agent and your model.

The agent, the sandbox and the task are chosen **per rollout**, so one server covers the whole
matrix. Adding a new agent does not mean writing a new environment.

### What you get back

Per model call:

```python
turn.prompt_token_ids       # the engine's own tokenisation of everything before this turn
turn.completion_token_ids   # what it sampled
turn.per_token_logps        # the behaviour-policy logprob of each sampled token
```

plus `result.reward` from the task's verifier.

Nothing is tokenised locally. The engine tokenises each prompt in order to serve it and returns
`prompt_token_ids`, so turn *k+1*'s prompt is by construction the canonical tokenisation of
everything before it, tool results included. Re-rendering a prompt offline with a chat template
drifts from what the model actually saw, and a prompt that differs by one token silently splits one
long conversation into several short ones.

## Prerequisites

You need an OpenAI-compatible endpoint, and it **must** be started with two flags:

```bash
vllm serve <model> --return-tokens-as-token-ids --logprobs-mode processed_logprobs
```

Without them the endpoint answers every request normally and returns no token ids. Rollouts look
perfect and contain nothing trainable. That failure has no loud edge, so the server checks for it at
startup and refuses to run rather than let it through.

Install the extra, which brings Harbor and every sandbox backend:

```bash
pip install "openenv[harbor]"      # needs Python 3.12 or newer
```

## Quick Start

### 1. See what this machine can do

```bash
openenv harbor info \
  --llm-url $LLM \
  --dataset AdithyaSK/data_agent_rl_environment_eval
```

```
llm       Qwen/Qwen3.5-9B  [ok]
sandboxes 2 of 4 usable
  [ok]   e2b
  [ok]   modal
  [--]   docker     Docker daemon is not running.
  [--]   daytona    SDK not installed (daytona).
datasets  1 split(s), 366 tasks
harnesses 16 validated of 30 known
```

Read-only, boots nothing. It tells you which sandboxes have working credentials **and** an
importable SDK, so you find out here rather than 90 seconds into a rollout.

### 2. Run one rollout, no server

```bash
openenv harbor rollout \
  --llm-url $LLM \
  --dataset AdithyaSK/data_agent_rl_environment_eval \
  --task-index 0 --harness opencode --sandbox e2b \
  --out rollout.json
```

```
[opencode / e2b] task 0: 0000_369_369503_qa_1 ...
   ok    reward=1.00  turns=9  roots=2  multi-turn  tokens=1043  atif=match  48s
```

This path involves no env server, which makes it the one to reach for when something breaks: if
`rollout` works and `serve` does not, the fault is in the serving layer and nothing below it.

### 3. Serve it

```bash
openenv harbor serve --llm-url $LLM --dataset org/train,org/eval
```

You get a Task API for discovery, one long-running `run_rollout` MCP tool, and a UI at `/web`.

```python
from harbor_env import HarborEnv

with HarborEnv(base_url="http://localhost:8000") as env:
    split = env.splits()[0]["name"]
    result = env.run_rollout(split=split, task_index=0, harness="opencode", sandbox="e2b")

    print(result.reward, result.n_turns)
    for turn in result.turns:
        print(len(turn.completion_token_ids), sum(turn.per_token_logps))
```

`harness` and `sandbox` are per call, so consecutive rollouts against the same server can use
different agents and different backends.

## Configuration

### CLI

| flag | meaning |
|---|---|
| `--llm-url` | OpenAI-spec endpoint. Required, no default, no environment fallback |
| `--dataset` | HF repo id, local directory, or Harbor `name@version`. Repeatable |
| `--harness` | a validated seam name, or `module:Class` for your own agent |
| `--sandbox` | Harbor environment type: `e2b`, `modal`, `docker`, ... |
| `--reward-key` | which reward key is the training signal, for multi-reward tasks |
| `--expose` | how the sandbox reaches the capture proxy: `gradio`, `cloudflare`, `direct` |
| `--keep-sandbox` | leave the sandbox alive for debugging |
| `--force-build` | rebuild the sandbox image, bypassing the content-hash cache |

`--llm-url` deliberately has no default. An unset endpoint produces rollouts that look completely
normal and carry no token ids, so it is better to fail immediately.

### Environment variables

Used when deploying, where there is no command line:

| variable | meaning |
|---|---|
| `OPENENV_LLM_URL` | OpenAI-spec endpoint. Required |
| `OPENENV_DATASETS` | comma-separated dataset specs |
| `OPENENV_MODEL` | served model id. Read from the endpoint when it serves exactly one |
| `E2B_API_KEY` | offer the `e2b` sandbox |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | offer the `modal` sandbox |
| `OPENAI_API_KEY` | some task verifiers use an LLM judge and need this |

## Environment Details

### Supported agents

16 harnesses are validated end to end, grouped by the wire dialect they speak:

| dialect | agents |
|---|---|
| chat-completions | `opencode`, `goose`, `qwen-coder`, `swe-agent`, `mini-swe-agent`, `openhands-sdk`, `openclaw`, `hermes`, `kimi-cli`, `pi`, `vibe`, `terminus-2` |
| OpenAI Responses | `codex`, `trae-agent` |
| Anthropic Messages | `claude-code` |
| Google generateContent | `gemini-cli` |

Supporting all four dialects rather than chat-completions alone is what makes the last four rows
work. Any other Harbor agent can be reached with `--harness module:Class`.

### Sandboxes and providers

Two different things share the word "sandbox", and mixing them up is a common early confusion:

- **Harbor backends** are where the *agent* runs. That is what `--sandbox` selects.
- **OpenEnv providers** (`local_docker`, `hf_sandbox`, `modal`, `aca`, ...) host the *env server*
  itself and have no `exec`.

A backend counts as usable only if its class imports **and** Harbor's own preflight passes.
Credentials alone are not enough: a provider with valid keys but no SDK installed would otherwise
report available and fail at rollout time.

### Rewards

The verifier produces a dictionary. OpenEnv wants one number. The dictionary is forwarded unchanged
and the scalar is chosen by an explicit rule:

1. one key, use it
2. a key named `reward`, use it
3. otherwise fail and ask for `--reward-key`

Combining several keys automatically would be inventing reward semantics, so it refuses instead.
All shaping belongs in the trainer.

**`reward=None` is not zero.** It means the verifier never ran. A dead sandbox scored as zero looks
like a wrong answer, which is how an infrastructure failure gets mistaken for a model result.

### Reading a result

| field | meaning |
|---|---|
| `ok` | the rollout is usable. `False` means something failed, and `error` says what |
| `reward` | the verifier's number, or `None` if it never ran |
| `n_turns` | model calls captured |
| `n_roots` | independent conversations. More than one means subagents or auxiliary calls |
| `turns[]` | per-call token ids, logprobs, text and tool calls |
| `conversations[]` | the full message list per conversation, system prompt included |
| `atif` | `match`, `MISMATCH`, or `none`. See below |
| `findings` | warnings worth reading before training on the rollout |

`atif` is an independent cross-check. Harbor's own trajectory file records what the *harness*
thought happened; the capture records what crossed the wire. Two measurements of the same rollout
through completely different paths. `match` means they agree call for call.

**A failed rollout returns a result, never an exception.** That is deliberate. In an in-process
design one exception on one rank hangs every rank at the distributed barrier, so behind an HTTP
boundary a failure has to come back as `ok=False` instead.

## Deploying to Hugging Face Spaces

```bash
openenv harbor push \
  --llm-url $LLM \
  --dataset org/train,org/eval \
  --repo-id you/harbor-env \
  --env-file .env
```

Configuration travels as Space variables, provider credentials as Space secrets. Add `--dry-run`
to print exactly what would be sent first, and `--recreate` to delete and redeploy for a clean test.

Two details that matter:

**Task suites are mounted, not downloaded.** A Harbor suite is thousands of small files and Space
disk is ephemeral, so a download is re-paid on every restart. `push` syncs the suites into a storage
bucket named after the Space and mounts it at `/data`. The copy is server side, and re-running
`push` copies only what is new.

**The Space must be public.** The capture proxy is served at `<space-url>/capture`, and a private
Space requires an auth header that the agent inside the sandbox does not send. This is safe because
the proxy rejects any caller without a registered session id, so a public mount is not an open
relay.

## Troubleshooting

**Rollout finishes with zero model calls.** The agent never reached the proxy. Usually the endpoint
URL is wrong, the model name did not resolve, or auth was rejected. Check the `findings` field,
which names the likely cause.

**`llm ... [FAILED]` at startup.** The endpoint cannot return token ids. Restart it with
`--return-tokens-as-token-ids --logprobs-mode processed_logprobs`.

**A sandbox shows `[--]` in `info`.** The detail column says why, and it is usually a missing
credential or a missing SDK. Install everything with `pip install "openenv[harbor]"`.

**`atif=none`.** That harness writes no trajectory file, so no cross-check is possible. Capture is
unaffected.

**Many roots for one rollout.** Normal for agents that run subagents or auxiliary calls. Each root
is a separate conversation, and only agent conversations are counted as trainable.

**Exit code 137.** The agent was killed inside the sandbox, almost always by the OOM killer on a
large input. That is a task failure, not a capture failure.

## How it works

```
agent in a sandbox
   |  base URL points at the proxy, API key is the capture session id
   v
capture proxy  --normalise to chat, force token ids on-->  your endpoint
   ^                                                            |
   |  replay in the agent's own dialect  <----------------------
   |
   +-- every call becomes a node in a rollout graph, linked by token prefix
```

The agent's API key is the capture session id, which is how one proxy serves many concurrent
rollouts without a port each.

Turns are linked by **exact token prefix**: a call whose `prompt_token_ids` begin with an existing
node's full sequence becomes its child. Nothing else is used, because request ids and timestamps are
per-agent and the prefix is not. That gives conversations, retries and subagent branches for free,
and lets abandoned branches be marked discarded so they are never trained with the reward the main
path earned.

Locally the proxy runs on its own port and is published to the sandbox. When hosted there is one
port and one public URL, so it is mounted on the env server's own app instead and nothing is
forwarded.

## References

- [Harbor](https://github.com/laude-institute/harbor), which provides the datasets, sandboxes,
  agents and verifiers
- [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
  ([paper](https://arxiv.org/abs/2605.24220)), the black-box approach this capture layer follows,
  and the source of the vendored dialect transformers
- [verifiers](https://github.com/willccbb/verifiers), whose `Dialect` model informed the auxiliary
  route and streaming handling
