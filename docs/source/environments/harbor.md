<!-- openenv-source: harbor_env -->
# harbor_env

Run a Harbor task with a coding agent and capture every token id and per-token logprob it produced,
ready to train on.

Harbor supplies the task datasets, sandbox backends, agents and verifiers. This environment adds the
OpenEnv surface: dataset discovery over the Task API, one `run_rollout` MCP tool, and a capture proxy
between the agent and your model.

## Configure

| variable | meaning |
|---|---|
| `OPENENV_LLM_URL` | OpenAI-spec endpoint (vLLM). **Required.** |
| `OPENENV_DATASETS` | comma-separated dataset specs — HF repo id, local dir, or Harbor `name@version` |
| `OPENENV_MODEL` | served model id; read from the engine when it serves exactly one |
| `E2B_API_KEY` | offer the `e2b` sandbox |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | offer the `modal` sandbox |

The engine **must** be started with:

```
--return-tokens-as-token-ids --logprobs-mode processed_logprobs
```

Without them it answers every request normally and returns no token ids, so captured rollouts are
empty and nothing reports an error. The server refuses to start rather than let that happen.

## Use

```python
from harbor_env import HarborEnv

with HarborEnv(base_url="http://localhost:8000") as env:
    split = env.splits()[0]["name"]
    result = env.run_rollout(split=split, task_index=0, harness="opencode", sandbox="e2b")
    print(result.reward, len(result.turns))
```

`harness` and `sandbox` are per-call, so consecutive rollouts can use different agents and different
backends against the same server.

## Notes

Two ports when run locally: the env server faces trainers and browsers, the capture proxy faces the
sandbox and is the only thing published. On a hosted platform there is one port and one public URL,
so the proxy is mounted on the env server's own app at `/capture` and nothing is forwarded. It still
refuses callers without a registered session id, which is what keeps a public mount from being an
open relay.

Capture quality and task success are independent. A rollout can be captured perfectly and score 0
because the model was wrong; a reward of 1 with unusable capture is worse than useless for training.
Both are reported separately.
