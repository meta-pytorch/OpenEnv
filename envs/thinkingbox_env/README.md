---
title: ThinkingBox Environment
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# ThinkingBox

## Introduction

[ThinkingBox](https://github.com/microsoft/thinkingbox) is an agentic harness for
defining isolated MCP tool environments, creating stateful scenarios and test
cases, running an LLM agent with simulated-user interaction, and evaluating
the resulting backend state, side effects, and response requirements. It can
support offline evaluation or conversation generation and agentic training
workflows.

This OpenEnv adapter exposes the public
[ThinkingBox-Bench](https://arxiv.org/abs/2608.19741) through the standard
WebSocket reset/step interface. The adapter is currently designed for
evaluation only.

## Quick Start

Start the externally managed ThinkingBox services described under
[Deployment Boundary](#deployment-boundary), then run the OpenEnv server:

```bash
OPENENV_TB_CONFIG=/path/to/thinkingbox.yaml \
uv run --project envs/thinkingbox_env --frozen server
```

Connect from a trusted harness:

```python
from thinkingbox_env import ThinkingBoxEnv

async with ThinkingBoxEnv(base_url="http://127.0.0.1:8000") as env:
    result = await env.reset("file.py:test_name")
    tools = await env.list_tools()
    result = await env.call_tool("tool_name", {"argument": "value"})
    result = await env.submit_message("Final response")
```

For native model orchestration, repetition IDs, privacy-reviewed canonical
JSONL, and strict aggregation, use
[`examples/thinkingbox/eval_testlist.py`](../../examples/thinkingbox/eval_testlist.py).

## Environment Details

The adapter preserves native ThinkingBox hydration, history replay, tool
overrides, parallel tool batches, direct responses, simulated users, effects,
fixtures, `TestContext`, `TestScript`, `Judge`, error latching, and teardown.
Final rewards are binary pass/fail; infrastructure failures are reported
separately and never converted into benchmark failures.

The model-facing action surface contains tool discovery, tool calls, and
assistant messages. A trusted harness may call `finish()`, but finish is not
advertised in the public action schema. Private user context, initialization
state, assertion source, credentials, proxy session identifiers, effects, and
grading internals remain inside the environment.

`seed` is accepted for OpenEnv API compatibility but intentionally unused
because tasks are selected deterministically by UID. Concurrent sessions are
stacked independent environment instances, one trajectory per instance; they
are not multiplexed trajectories.

Executable assets come from the pinned `thinkingbox-bench-v1.0` release in
[`microsoft/thinkingbox-data`](https://github.com/microsoft/thinkingbox-data).
The [Hugging Face `microsoft/ThinkingBox-Bench`
dataset](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench) is a
viewer-friendly representation, while GitHub remains the executable source.

## Configuration

The server accepts an ordinary native ThinkingBox YAML configuration through
`OPENENV_TB_CONFIG` or the reset `config=` argument. The file supplies
`mcp_proxy`, `orchestrator.agent_model`, `user_model`, `judge_model`,
`judge_type`, and user-completion behavior; see the
[ThinkingBox LLM configuration
guide](https://github.com/microsoft/thinkingbox/blob/main/docs/llm_endpoint_config.md).

| Variable | Default | Purpose |
|---|---|---|
| `OPENENV_TB_PROXY_URL` | `http://127.0.0.1:7111` | Session Proxy fallback |
| `OPENENV_TB_DATASET` | pinned release | Local executable-data override |
| `OPENENV_TB_DATA_CACHE` | `~/.cache/openenv/thinkingbox_bench` | Verified release cache |
| `OPENENV_TB_AGENT` | `think` | Native agent definition |
| `OPENENV_TB_CONFIG` | unset | Native server-side model/proxy config |
| `OPENENV_TB_PROXY_TIMEOUT` | `120` | Proxy timeout without config override |
| `OPENENV_TB_DATA_TIMEOUT` | `120` | Release download timeout |
| `OPENENV_TB_MAX_CONCURRENT_ENVS` | `8` | Independent environment instances |

The client message timeout remains longer than server-side model and episode
operations so a valid long-running turn is not abandoned prematurely.

## Deployment Boundary

The image starts only the OpenEnv API. It currently needs an externally
managed ThinkingBox Session Proxy, benchmark MCP servers, Typesense, and agent,
user-simulator, and judge model endpoints. `GET /health` reports process
liveness; `GET /ready` reports the observable external dependencies and marks
scenario-specific Typesense readiness as unobservable from this process.

## Citation

Please cite the paper if ThinkingBox is useful in your experiments.

- Paper: [One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark
  for Agents in Stateful Business Workflows](https://arxiv.org/abs/2608.19741)
- Framework: [`microsoft/thinkingbox`](https://github.com/microsoft/thinkingbox)
- Executable benchmark:
  [`microsoft/thinkingbox-data`](https://github.com/microsoft/thinkingbox-data)
- Viewer dataset:
  [`microsoft/ThinkingBox-Bench`](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench)

```bibtex
@misc{li2026successisntreliabilitythinkingbox,
  title={One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows},
  author={Zhuochun Li and Youngmin Ko and Ali Keramati and Nicola Ferri and Susana Palmaz Lopez Pelaez and Liang-Chun Tsai and Calvin Wang and Mirco Milletari and Tuhin Kundu and Vadim Smolyakov and Kjartan Olafsson and Tommy Guy},
  year={2026},
  eprint={2608.19741},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.19741},
}
```
