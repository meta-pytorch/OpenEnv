<!-- openenv-source: claude_code_env -->
# Claude Code Environment for OpenEnv

`claude_code_env` runs the [Claude Code](https://github.com/anthropics/claude-code)
CLI (`@anthropic-ai/claude-code`) inside an isolated
[Hugging Face sandbox](https://huggingface.co/docs/huggingface_hub/package_reference/sandbox)
against an OpenAI-compatible LLM endpoint, capturing per-token logprobs for GRPO
training.

It is a sibling of `opencode_env` and `pi_env`: same two-layer design (an
in-process harness primitive plus a deployable HTTP env), same transparent-proxy
logprob capture, same uniform `(instruction, setup, verify)` Task shape. The
agent is Claude Code, and the default sandbox backend is Hugging Face.

The env is **task-agnostic**. Every rollout is configured at call-time with a
uniform Task shape:

  - **`instruction`**: prompt for the agent
  - **`setup`**: list of bash commands run *before* the agent (pip install, git
    clone, file downloads, anything you need staged in the sandbox)
  - **`verify`**: list of bash commands run *after* the agent (asserts, pytest
    invocations, score-file writes)

Reward = `passed_verify / total_verify` unless any `verify` command writes a
float to `/root/logs/verifier/reward.txt` (override).

## How it works: the Anthropic translation shim

Claude Code speaks the **Anthropic Messages API only**, so in `transparent_proxy`
mode an in-sandbox **translation shim** (`anthropic_shim.py`) sits in front of the
shared interception proxy:

```
Claude Code --(Anthropic /v1/messages)--> shim --(OpenAI /v1/chat/completions)--> interception proxy --> vLLM
```

The shim is a pure translator. It converts each Anthropic Messages request into
an OpenAI chat-completions request, forwards it unary to the interception proxy,
and translates the reply back into the Anthropic shape (replaying it as a
synthetic Anthropic SSE sequence when Claude Code asked to stream). The proxy
injects `logprobs=true` and captures `completion_token_ids` + `per_token_logps`
on the OpenAI/vLLM side, exactly as it does for `opencode_env` and `pi_env`.
Training correctness is unchanged: the shim only translates the envelope, and
capture still happens at the vLLM seam.

Serve the upstream vLLM with a generous `--max-model-len`. Claude Code's system
prompt is large and grows each turn, so `prompt_tokens + max_tokens` can exceed a
small context window and the upstream returns 400. `proxy_max_tokens_cap` (default
`8192`) bounds the completion side, but the server still needs headroom for the
prompt (`--max-model-len 98304` works well for Qwen3-4B).

Inside the sandbox the agent runs headless via `claude -p --output-format json
--dangerously-skip-permissions`, with the prompt piped on stdin. Claude Code is
pointed at the shim (or, in `black_box` mode, at an Anthropic-native endpoint)
via `ANTHROPIC_BASE_URL`. Its config dir is `.claude` (a `settings.json` marks
onboarding complete so `claude -p` does not block on first-run prompts). The
sandbox execs as root, so `IS_SANDBOX=1` is set to let
`--dangerously-skip-permissions` run. On the first rollout the sandbox bootstraps
Node 22 and installs `@anthropic-ai/claude-code`. The sandbox layer (backend plus
interception proxy) comes from `openenv.core.sandbox`.

## In-process primitive (no HTTP)

For trainers that drive a sandbox directly without an HTTP boundary. This is what
loop-owning GRPO training uses. The primitive uses the sandbox backend + proxy
from `openenv.core.sandbox`:

```python
import os
from claude_code_env import ClaudeCodeConfig, ClaudeCodeSessionFactory, ClaudeCodeTask, HFSandboxBackend

factory = ClaudeCodeSessionFactory(
    config=ClaudeCodeConfig(
        base_url="https://my-vllm-endpoint/v1",       # the real upstream the proxy forwards to
        api_key=os.environ.get("VLLM_API_KEY", "intercepted"),
        model="Qwen/Qwen3.5-4B",
        sandbox_home="/root",                          # HF sandbox execs as root
    ),
    sandbox_backend=HFSandboxBackend(image="python:3.12"),
    mode="transparent_proxy",                          # shim + proxy capture per-token logprobs
)
session = factory.create(task=ClaudeCodeTask(instruction="..."))
session.wait_for_completion()
turns = session.fetch_proxy_trace()                    # per-turn (tokens, logprobs)
session.close()
```

In `transparent_proxy` mode the factory starts the interception proxy on
`localhost:7000` (pointed at `base_url`), then starts the Anthropic-to-OpenAI shim
on `localhost:7100` in front of it, and points Claude Code at the shim via
`ANTHROPIC_BASE_URL`. Each entry returned by `fetch_proxy_trace()` carries
`request`, `response`, `completion_tokens`, `completion_token_ids`,
`per_token_logps`, `finish_reason`, and `latency_s`.

### Sandbox backend

`HFSandboxBackend` (from `openenv.core.sandbox`) runs the agent in a Hugging Face
sandbox. `image="python:3.12"` cold-installs Node 22, the Claude Code CLI
(`npm install -g @anthropic-ai/claude-code`), and the proxy's Python deps on every
rollout. For faster rollouts use the pre-baked image (Node + Claude Code + proxy deps
already installed), built by CI from `hf_image/Dockerfile`:

```python
sandbox_backend=HFSandboxBackend(image="ghcr.io/huggingface/openenv-claude-code-sandbox:latest")
```

Any backend satisfying the `SandboxBackend` / `SandboxHandle` / `BgJob` protocols
in `openenv.core.sandbox.base` can be plugged in the same way.

## Deployed env (HTTP)

The deployed Space exposes:

- **Web UI** at `/web`: pick endpoint, write task, hit Run, watch live phase log
  + reward + logprobs.
- **MCP tool API** at `/mcp`: programmatic `run_rollout` calls.
- **OpenAPI docs** at `/docs`, **health** at `/health`.

```python
import os
from claude_code_env import ClaudeCodeEnv

with ClaudeCodeEnv(base_url="https://<user>-claude-code-env.hf.space") as env:
    env.reset()
    result = env.run_rollout(
        endpoint="openai",                          # vllm | openai | hf_router
        api_key=os.environ["OPENAI_API_KEY"],       # or set as a Space secret
        instruction=(
            "Create binary_search.py exposing def binary_search(arr, target) -> int "
            "that returns the index of target in arr, or -1 if absent."
        ),
        setup=[],
        verify=[
            "test -f /root/workdir/binary_search.py",
            "python -c \"import sys; sys.path.insert(0, '/root/workdir'); "
            "import binary_search; "
            "assert binary_search.binary_search([1,2,3], 2) == 1; print('OK')\"",
        ],
        task_id="binary_search_v1",
    )
    print("reward:", result.reward)
    print("turns:", len(result.proxy_turns))
```

## The MCP Tool: `run_rollout`

Single tool, two ways to specify the LLM endpoint:

**Option A: endpoint shorthand (recommended)**: pass `endpoint="vllm"` (or
`"openai"` / `"hf_router"`). The server resolves `base_url`, `api_key`, and
`model` from env vars + catalog defaults. Any explicit field overrides.

**Option B: fully explicit**: pass `base_url` + `api_key` + `model` directly.

| Arg | Type | Default | Notes |
|---|---|---|---|
| `endpoint` | `str` | `""` | One of `"vllm"` / `"openai"` / `"hf_router"`. |
| `base_url` / `api_key` / `model` | `str` | `""` | Override / supply explicitly. |
| `instruction` | `str` | required | Prompt passed to `claude`. |
| `setup` | `list[str]` | `[]` | Bash commands run **before** the agent. |
| `verify` | `list[str]` | `[]` | Bash commands run **after** the agent. |
| `task_id` | `str` | `""` | Echoed back in result. |
| `mode` | `str` | `"transparent_proxy"` | Or `"black_box"` (no logprobs). |
| `disable_thinking` | `bool \| None` | `None` (catalog default) | Inject `chat_template_kwargs.enable_thinking=false`. |
| `max_tokens_cap` | `int` | `4096` | Per-turn `max_tokens` clamp. |
| `top_logprobs` | `int` | `5` | HF Router cap is 5, OpenAI 0 to 20, vLLM unbounded. |
| `agent_timeout_s` | `float` | `900.0` | Hard wall budget for one `claude` run. |
| `image` | `str` | `""` | HF sandbox image. Blank falls back to `python:3.12` (cold-installs Node + Claude Code). |

Returns `RolloutResult` JSON with: `reward`, `setup_results[]`,
`verify_results[]`, `proxy_turns[]`, `files{}`, `agent_log_tail`,
`proxy_log_tail`, `wall_s`, `agent_exit_code`, `sandbox_id`, `error`.

## Two Operating Modes

| Mode | What it does | Best for |
|---|---|---|
| **`transparent_proxy`** (default) | The in-sandbox shim (`localhost:7100`) translates Claude Code's Anthropic calls to OpenAI and forwards to the interception proxy (`localhost:7000`), which injects `logprobs=true` and captures per-turn `(messages, completion_token_ids, per_token_logps)` to `proxy_trace.jsonl`. | GRPO / RL training, observability, top-k distillation. |
| **`black_box`** | No shim, no proxy. Claude Code talks straight to an Anthropic-native `base_url`. | Smoke tests, eval, SFT data collection. |

## Building the Docker Image

```bash
cd envs/claude_code_env

openenv validate                    # check pyproject.toml + openenv.yaml + server/app.py + uv.lock
openenv build -t claude-code-env    # builds the image (uses server/Dockerfile)

# run locally with an HF token (Sandbox access)
docker run -p 8000:8000 -e HF_TOKEN=hf_... claude-code-env
```

Or build directly:

```bash
docker build -t claude-code-env -f envs/claude_code_env/server/Dockerfile envs/claude_code_env
```

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | **yes** for any rollout | Hugging Face sandbox credentials. |
| `MAX_CONCURRENT_ENVS` | no | Env-instance pool size. Default `4`. |
| `ENABLE_WEB_INTERFACE` | no | Set `false` to disable the `/web` Gradio mount. Default `true`. |
| `VLLM_URL` / `VLLM_API_KEY` / `VLLM_MODEL` | for `endpoint="vllm"` | OAI-compatible base URL (key defaults to `intercepted`). |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | for `endpoint="openai"` | Standard OpenAI. |
| `HF_ROUTER_API_KEY` / `HF_ROUTER_BASE_URL` / `HF_ROUTER_MODEL` | for `endpoint="hf_router"` | HF Router. |

Pick `provider:` suffixes that actually return logprobs:
**Together / Nscale / Scaleway / SambaNova / Cerebras**. Avoid Novita /
Hyperbolic / Featherless (silent drop) and Groq (HTTP 400).

## Project Structure

```
claude_code_env/
├── README.md                       # this file
├── openenv.yaml                    # OpenEnv space spec
├── pyproject.toml                  # deps + ``server`` entrypoint
├── __init__.py                     # re-exports primitive + client + models
│
├── client.py                       # ClaudeCodeEnv(MCPToolClient)
├── models.py                       # RolloutResult / RolloutTurn / ClaudeCodeState
│
├── config.py                       # ClaudeCodeConfig (primitive)
├── harness.py                      # ClaudeCodeSession / ClaudeCodeSessionFactory (CLI-only)
├── claude_code_runtime.py          # settings.json builder + install/run cmds
├── anthropic_shim.py               # Anthropic Messages -> OpenAI translation shim
├── task.py                         # ClaudeCodeTask
│
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI factory, mounts Gradio at /web
    ├── claude_code_environment.py  # MCPEnvironment with single ``run_rollout`` tool
    ├── gradio_ui.py                # the /web Gradio Blocks UI
    ├── catalog.py                  # endpoint shorthand resolver
    └── Dockerfile                  # multi-stage uv build (used by ``openenv build``)
```

The sandbox backend + interception proxy are imported from
`openenv.core.sandbox`. `claude_code_env` ships only the Anthropic translation
shim (`anthropic_shim.py`) on top.
