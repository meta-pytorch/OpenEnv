# OpenCode Harness Primitive

A reusable harness primitive for running the [OpenCode](https://opencode.ai/)
CLI coding agent inside an isolated sandbox against any OpenAI-compatible
endpoint. Implements the `ResourceSession` / `ResourceSessionFactory` contracts
from `openenv.core.harness` (RFC 005 / PR #471).

## Why

OpenCode is a closed binary: it owns its own agent loop, tool-calling, and
file I/O, and it speaks to an LLM only over the OpenAI chat-completions API.
This package wraps that loop behind OpenEnv's harness interface so a trainer
or evaluator can:

- spawn one isolated sandbox per rollout (E2B by default),
- point opencode at any OpenAI-compatible endpoint,
- optionally intercept every LLM request to capture per-token logprobs for
  on-policy RL training,
- run a user-supplied verifier to produce a scalar reward.

## Operating modes

| Mode | Endpoint opencode hits | What the proxy does | Trainer receives | GRPO-ready? |
|---|---|---|---|---|
| `black_box` | the real LLM URL | no proxy | transcript + reward | no — eval / SFT only |
| `transparent_proxy` (default) | in-sandbox proxy that forwards to the real URL | injects `logprobs=true`, captures per-turn messages + completion tokens + logprobs | per-turn `(messages, completion_tokens, logprobs, reward)` | yes (on-policy in colocate serve/async) |

## Installation

The package lives under `OpenEnv/envs/opencode_env/`. Install directly from git:

```toml
[project.dependencies]
openenv-opencode_env = {
  git = "https://github.com/adithya-s-k/OpenEnv.git",
  branch = "opencode-harness",
  subdirectory = "envs/opencode_env"
}
```

Or via `uv add`:

```bash
uv add "openenv-opencode_env @ git+https://github.com/adithya-s-k/OpenEnv.git@opencode-harness#subdirectory=envs/opencode_env"
```

When PR #471 merges into `meta-pytorch/OpenEnv:main`, this branch will be
rebased onto `main` and upstreamed.

## Quick start — Mode A (black-box) against real OpenAI

```python
import os
from openenv.core.harness import VerifyResult
from opencode_env import OpenCodeConfig, OpenCodeSessionFactory, E2BSandboxBackend

def my_verifier(sandbox, task):
    r = sandbox.exec("cd /home/user/workdir && python fizzbuzz.py | head -5")
    reward = 1.0 if "Fizz" in r.stdout else 0.0
    return VerifyResult(env_reward=reward, done=True, artifacts={"stdout": r.stdout})

factory = OpenCodeSessionFactory(
    config=OpenCodeConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="openai/gpt-4o-mini",
    ),
    sandbox_backend=E2BSandboxBackend(),  # reads E2B_API_KEY
    mode="black_box",
    verifier=my_verifier,
)

session = factory.create(task={"instruction": "write fizzbuzz.py in the cwd"})
try:
    session.wait_for_completion(timeout_s=180)
    result = session.verify(transcript=[])
    print("reward:", result.env_reward)
finally:
    session.close()
```

## Quick start — Mode B (transparent proxy) against a tunneled vLLM

```python
factory = OpenCodeSessionFactory(
    config=OpenCodeConfig(
        provider="openai_compatible",
        base_url=os.environ["VLLM_TUNNEL_URL"] + "/v1",
        api_key="intercepted",
        model="openai_compatible/Qwen/Qwen3.5-4B",
        proxy_disable_thinking=True,        # Qwen3/Qwen3.5 tokenizer hook
        proxy_max_tokens_cap=4096,
    ),
    sandbox_backend=E2BSandboxBackend(),
    mode="transparent_proxy",
    verifier=my_verifier,
)
session = factory.create(task={"instruction": "..."})
session.wait_for_completion()

# fetch_proxy_trace returns a list of dicts, one per LLM turn,
# each with messages, completion_tokens, completion_token_ids,
# per_token_logps, finish_reason, latency_s
turns = session.fetch_proxy_trace()
```

## Post-rollout summary

```python
from opencode_env import collect_rollout_summary, print_rollout_summary

summary = collect_rollout_summary(session)
print_rollout_summary(summary)   # turns, tokens, tool calls, errors, workdir files
```

## `OpenCodeTask`

The primitive accepts a bare string, a dict, or a typed `OpenCodeTask`:

```python
from opencode_env import OpenCodeTask

OpenCodeTask(
    instruction="Build a REST API with Flask",
    setup_shell="pip install flask",                     # runs once before opencode
    upload_files={"/home/user/workdir/schema.json": "..."},  # staged into sandbox
    metadata={"task_id": "flask_api_v1"},                # passed through to verifier
)
```

## Config knobs

`OpenCodeConfig` fields (see `config.py` for full list):

| field | default | notes |
|---|---|---|
| `provider` | `"openai_compatible"` | `"openai"`, `"anthropic"`, or `"openai_compatible"` |
| `base_url` | required | provider endpoint (`https://host/v1`) |
| `api_key` | `"intercepted"` | provider key; vLLM ignores |
| `model` | `"intercepted/model"` | `<provider>/<model>` form; override per provider |
| `opencode_version` | `"latest"` | pin via e.g. `"0.5.3"` |
| `disabled_tools` | `["webfetch", "question"]` | see opencode docs |
| `system_prompt` | `None` | uploaded to sandbox, used by opencode |
| `sandbox_home` | `"/home/user"` | override for non-E2B backends |
| `proxy_max_tokens_cap` | `16384` | Mode B: clamp `max_tokens`/`max_completion_tokens` |
| `proxy_top_logprobs` | `5` | Mode B: top-k per token |
| `proxy_disable_thinking` | `False` | Mode B: inject `chat_template_kwargs.enable_thinking=false` (Qwen3/Qwen3.5) |

## Sandbox backends

`E2BSandboxBackend` is the default. Any class implementing the
`SandboxBackend` Protocol (`sandbox/base.py`) works:

```python
class SandboxBackend(Protocol):
    def create(self, *, timeout_s: int, envs, metadata) -> SandboxHandle: ...

class SandboxHandle(Protocol):
    sandbox_id: str
    def exec(self, cmd, *, envs=None, cwd=None, timeout=60) -> ExecResult: ...
    def start_bg(self, cmd, *, envs=None, cwd=None) -> BgJob: ...
    def write_text(self, path, content): ...
    def read_text(self, path) -> str: ...
    def exists(self, path) -> bool: ...
    def kill(self): ...
```

## Known limitations

- **OpenAI's gpt-5.x chat family refuses logprob requests** (HTTP 403), so
  Mode B against OpenAI requires `gpt-4o-mini` or older. vLLM-hosted models
  return logprobs natively — the intended training path.
- The transparent proxy currently speaks only `/v1/chat/completions`. OpenAI's
  newer `/v1/responses` endpoint (used by `@ai-sdk/openai`) is not supported;
  the factory auto-switches the provider to `@ai-sdk/openai-compatible` in
  Mode B for this reason.
- E2B's `commands.run(background=True)` has a default server-side
  `timeout=60s`; we override it to `0` for long-running agent jobs.

## Tests

```bash
# Unit tests only (fast, no sandbox)
PYTHONPATH=src:envs/opencode_env uv run pytest envs/opencode_env/tests/ -q \
    --ignore=envs/opencode_env/tests/test_sandbox_e2b.py \
    --ignore=envs/opencode_env/tests/test_harness_live_openai.py \
    --ignore=envs/opencode_env/tests/test_harness_live_mode_b.py \
    --ignore=envs/opencode_env/tests/test_harness_live_vllm.py

# Live E2B sandbox integration
E2B_API_KEY=... PYTHONPATH=src:envs/opencode_env uv run pytest \
    envs/opencode_env/tests/test_sandbox_e2b.py -v

# End-to-end Mode A against real OpenAI
E2B_API_KEY=... OPENAI_API_KEY=... PYTHONPATH=src:envs/opencode_env uv run pytest \
    envs/opencode_env/tests/test_harness_live_openai.py -v -s

# End-to-end Mode B against a tunneled vLLM
E2B_API_KEY=... VLLM_TUNNEL_URL=https://xxx.trycloudflare.com/v1 \
    VLLM_MODEL=Qwen/Qwen3.5-4B \
    PYTHONPATH=src:envs/opencode_env uv run pytest \
    envs/opencode_env/tests/test_harness_live_vllm.py -v -s
```

## References

- [OpenEnv PR #471](https://github.com/meta-pytorch/OpenEnv/pull/471) — harness session runtime we stack on
- [OpenCode docs](https://opencode.ai/docs/) — CLI, config, providers
- [E2B sandbox SDK](https://e2b.dev/docs)
