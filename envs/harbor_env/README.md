---
title: Harbor
emoji: ⚓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# Harbor Environment

**Train one policy against many coding agents.** Pick a Harbor dataset, pick an agent, pick a
sandbox, and get back the exact token ids and per-token logprobs of every model call the agent made,
plus the task's own reward.

## Overview

An agent harness is a moving part you probably do not want to own. opencode, codex, claude-code and
gemini-cli each have their own loop, their own tool surface and their own wire format, and a policy
trained against exactly one of them learns that one's habits.

The usual cost of supporting several is one integration per agent. Here it is one integration total:

| | |
|---|---|
| **16 harnesses** | validated end to end, across 4 wire dialects |
| **23 sandbox backends** | from Harbor, 4 with credential checks wired in |
| **any Harbor dataset** | HF repo, local directory, or Harbor registry name |

All three are chosen **per rollout**, so one server covers the whole matrix and rotating the harness
during training is a config change rather than a new environment.

Harbor supplies the tasks, sandboxes, agents and verifiers. This environment adds the OpenEnv
surface: dataset discovery over the Task API, one `run_rollout` MCP tool, a capture proxy, and a UI.

### What you get back

Per model call:

```python
turn.prompt_token_ids       # the engine's own tokenisation of everything before this turn
turn.completion_token_ids   # what it sampled
turn.per_token_logps        # the behaviour-policy logprob of each sampled token
```

plus `result.reward` from the task's verifier. That tuple is the whole training contract.

## The intercept

The agent is a black box. It is a real CLI tool running in a sandbox, and it was never written with
training in mind. So instead of modifying it, an OpenAI-spec proxy is placed between it and your
model, and every call is recorded as it passes.

```
agent in a sandbox
   |  base URL points at the proxy, API key IS the capture session id
   v
capture proxy  --normalise to chat, force token ids on-->  your endpoint
   ^                                                            |
   |  replay in the agent's own dialect  <----------------------+
   |
   +-- every call becomes a node in a rollout graph, linked by token prefix
```

Three properties make this work across agents rather than for one:

**Nothing is tokenised locally.** The engine tokenises each prompt in order to serve it and hands
back `prompt_token_ids`, so turn *k+1*'s prompt is by construction the canonical tokenisation of
everything before it, tool results included. Re-rendering a prompt offline with a chat template
drifts from what the model actually saw, and a prompt that differs by one token silently splits one
long conversation into several short ones.

**Four wire dialects.** Coding agents did not converge on one API. chat-completions, OpenAI
Responses, Anthropic Messages and Google `generateContent` are all translated to a single upstream
shape and replayed in the dialect the agent expects, streaming included. That is what makes codex,
claude-code and gemini-cli work rather than only the chat-completions agents.

**The API key is the session id.** One proxy serves many concurrent rollouts with no port each, and
a caller without a registered session is rejected, so the proxy is safe to expose to a sandbox.

Turns are linked into a graph by **exact token prefix**: a call whose `prompt_token_ids` begin with
an existing node's full sequence becomes its child. Nothing else is consulted, because request ids
and timestamps are per-agent and the prefix is not. Conversations, retries and subagent branches
fall out of that for free, and a branch the agent abandoned is marked discarded so it is never
trained with the reward the main path earned.

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

## CLI reference

Four commands. Every flag below is the complete set, with its type and default. `openenv harbor
<command> --help` prints the same thing.

Exit codes:

| code | meaning |
|---|---|
| `0` | success |
| `1` | ran, but failed. `rollout` returns this if **any** rollout in the batch was unusable |
| `2` | usage error: a missing or invalid flag. Nothing ran |

The `1` and `2` split matters if you are scripting this: `2` means the command never started, so
retrying it unchanged will fail the same way.

### `openenv harbor info`

Report what this machine can run. Read-only: boots no sandbox, starts no server, makes no rollout.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | `""` | OpenAI-spec endpoint. Optional here; without it the LLM section is skipped and the rest still reports |
| `--model` | str | `""` | Served model id. Auto-detected when the endpoint serves exactly one |
| `--dataset` | str | none | Dataset spec. Repeatable, or comma-separated |
| `--env-file` | path | `""` | dotenv with provider credentials, loaded before the checks |
| `--verbose` | flag | off | List all 30 harnesses, not only the 16 validated |
| `--json` | flag | off | Emit machine-readable JSON instead of the text report |

```bash
openenv harbor info --llm-url $LLM --dataset org/tasks --json
```

The JSON has four top-level keys: `llm`, `sandboxes`, `datasets`, `harnesses`. Each sandbox entry is
`{name, available, detail}`, so a script can select a backend without parsing prose:

```bash
openenv harbor info --llm-url $LLM --json \
  | jq -r '.sandboxes[] | select(.available) | .name'
```

### `openenv harbor rollout`

Run rollouts with no env server involved. This is the debugging path: if `rollout` works and `serve`
does not, the fault is in the serving layer and nothing below it.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | **required** | OpenAI-spec endpoint. No default and no env fallback, on purpose |
| `--dataset` | str | **required** | Dataset spec. Only the first is used by this command |
| `--task-index` | int | `0` | Index into the split. Stable: index is a task's identity |
| `-n`, `--n-tasks` | int | `1` | Run this many consecutive tasks from `--task-index` |
| `--harness` | str | `opencode` | A validated seam name, or `module:Class` for your own agent |
| `--sandbox` | str | `e2b` | Harbor environment type |
| `--model` | str | `""` | Served model id. Auto-detected when unambiguous |
| `--port` | int | `8100` | Local port for the capture proxy. One per concurrent process |
| `--expose` | str | `gradio` | How the sandbox reaches the proxy: `gradio`, `cloudflare`, `direct` |
| `--reward-key` | str | `""` | Which reward key is the training signal, for multi-reward tasks |
| `--trials-dir` | path | tmp | Where Harbor writes trial artifacts |
| `--keep-sandbox` | flag | off | Leave sandboxes alive for debugging |
| `--force-build` | flag | off | Rebuild the sandbox image, bypassing the content-hash cache |
| `--env-file` | path | `""` | dotenv with provider credentials |
| `--out` | path | `""` | Write the full result JSON, token ids and logprobs included |

```bash
openenv harbor rollout --llm-url $LLM --dataset org/tasks \
  --task-index 0 -n 5 --harness codex --sandbox modal --out results.json
```

`-n` runs tasks sequentially in one process, reusing one proxy and one forward. To parallelise, run
several processes and **give each its own `--port`**. Two processes sharing a port is refused with
an error naming the process that holds it.

Use `--force-build` when a task has never been built on this account, or when a cached image has
drifted because the task pins its dependencies loosely.

### `openenv harbor serve`

Start the env server: Task API for discovery, one long-running `run_rollout` MCP tool, and a UI at
`/web`.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | **required** | OpenAI-spec endpoint |
| `--dataset` | str | none | Dataset specs to serve as splits. Repeatable |
| `--model` | str | `""` | Served model id |
| `--host` | str | `0.0.0.0` | Bind address |
| `--port` | int | `8000` | Env server port. Faces the trainer and the browser |
| `--capture-port` | int | `8100` | Capture proxy port. Faces the sandbox |
| `--expose` | str | `gradio` | How the sandbox reaches the proxy |
| `--env-file` | path | `""` | dotenv with provider credentials |

Refuses to start if the endpoint cannot return token ids.

### `openenv harbor push`

Deploy the same server to a Hugging Face Space.

| flag | type | default | meaning |
|---|---|---|---|
| `--llm-url` | str | **required** | Endpoint the deployed Space will use |
| `--repo-id` | str | **required** | Target Space, e.g. `you/harbor-env` |
| `--dataset` | str | none | Dataset specs. Repeatable |
| `--model` | str | `""` | Served model id |
| `--bucket` | str | Space name | Storage bucket holding the task suites. `none` disables the mount and downloads instead |
| `--hardware` | str | `""` | Space hardware, e.g. `cpu-basic` |
| `--private` | flag | off | Create it private. Rollouts then cannot work, see below |
| `--recreate` | flag | off | Delete the Space first, then deploy fresh |
| `--dry-run` | flag | off | Print exactly what would be sent and stop |
| `--env-file` | path | `""` | dotenv whose provider keys become Space **secrets** |

```bash
openenv harbor push --llm-url $LLM --dataset org/train,org/eval \
  --repo-id you/harbor-env --env-file .env --dry-run
```

`--private` is supported but rollouts will not work on a private Space: the capture proxy is served
at `<space-url>/capture`, and a private Space requires an auth header the sandboxed agent does not
send. Use it only to park a deployment.

## Supported harnesses

16 of the 30 known agents are validated end to end. "Validated" means a real rollout produced token
ids and logprobs, and where the agent emits a trajectory, its own record agreed with the capture.

| harness | dialect | runs |
|---|---|---|
| `opencode` | chat-completions | in sandbox |
| `goose` | chat-completions | in sandbox |
| `qwen-coder` | chat-completions | in sandbox |
| `swe-agent` | chat-completions | in sandbox |
| `mini-swe-agent` | chat-completions | in sandbox |
| `openhands-sdk` | chat-completions | in sandbox |
| `openclaw` | chat-completions | in sandbox |
| `hermes` | chat-completions | in sandbox |
| `kimi-cli` | chat-completions | in sandbox |
| `pi` | chat-completions | in sandbox |
| `vibe` | chat-completions | in sandbox |
| `terminus-2` | chat-completions | **host side** |
| `codex` | OpenAI Responses | in sandbox |
| `trae-agent` | OpenAI Responses | in sandbox |
| `claude-code` | Anthropic Messages | in sandbox |
| `gemini-cli` | Google generateContent | in sandbox |

Supporting four dialects rather than chat-completions alone is what makes the last four rows work.

`terminus-2` runs in the server process rather than inside the sandbox, so it reaches the proxy on
localhost and needs no public URL.

The other 14 known agents have a seam but are untested; run `openenv harbor info --verbose` to list
them. Anything Harbor supports can be reached with `--harness module:Class`.

## Supported sandboxes

`openenv harbor info` checks these four by default and reports why any is unusable:

| sandbox | credentials | validated |
|---|---|---|
| `e2b` | `E2B_API_KEY` | yes, extensively |
| `modal` | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, or `~/.modal.toml` | yes, extensively |
| `docker` | none, but the daemon must be running | works, not swept |
| `daytona` | `DAYTONA_API_KEY`, or `DAYTONA_JWT_TOKEN` + `DAYTONA_ORGANIZATION_ID` | not swept |

e2b and modal were compared on identical tasks and came out indistinguishable, which is the check
that matters: a backend-specific capture bug is exactly what a single-backend test hides.

Harbor registers 23 backends in total. Any of them can be passed to `--sandbox`; the four above are
the ones with a credential check wired in.

## Environment details

### Where the proxy runs

Locally there are two ports: the env server faces the trainer and the browser, the capture proxy
faces the sandbox and is the only one published. Sharing one port would expose the env server the
moment the proxy became reachable.

Hosted, that inverts. A Space has one port and one public URL, so the proxy is mounted on the env
server's own app at `/capture` and nothing is forwarded. It still rejects callers without a
registered session id, which is what keeps a public mount from being an open relay.

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

## References

- [Harbor](https://github.com/laude-institute/harbor), which provides the datasets, sandboxes,
  agents and verifiers
- [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
  ([paper](https://arxiv.org/abs/2605.24220)), the black-box approach this capture layer follows,
  and the source of the vendored dialect transformers
- [verifiers](https://github.com/willccbb/verifiers), whose `Dialect` model informed the auxiliary
  route and streaming handling
