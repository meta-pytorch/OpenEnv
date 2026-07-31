# Training a Real Coding Agent with GRPO (OpenCode)

This tutorial covers the black-box training path: training the actual
[`opencode`](https://opencode.ai) coding agent, with its own planner, tools,
context management, and stop condition, using TRL's experimental
`AsyncGRPOTrainer`. The agent owns its loop, and OpenEnv captures what it did.

> [!NOTE]
> Three GRPO patterns, three tutorials. For a standard `reset()` / `step()`
> flow where TRL drives the episode, see the
> [Wordle GRPO tutorial](wordle-grpo.md). For harness rollouts where the
> trainer still generates each turn (white-box), see the
> [BrowserGym harness tutorial](browsergym-harness.md). Use this page when you
> want to train a production agent as-is, without reimplementing its loop.

## How It Works

The full recipe lives in TRL. The moving pieces:

1. Each rollout runs the agent inside an OpenEnv session created by
   `OpenCodeSessionFactory` from
   [`opencode_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/opencode_env),
   in `transparent_proxy` mode. A small proxy inside the sandbox forwards the
   agent's `/v1/chat/completions` calls to your vLLM server and records each
   turn's token ids and logprobs to a trace.
2. When the agent stops, TRL's `HarnessRolloutWorker` reads the trace, rebuilds
   the per-turn training rows from the recorded ids, and scores the final
   workspace with the session's `verify()` method (a held-out verifier the
   agent never sees).
3. `AsyncGRPOTrainer` trains on those rows, propagating the rollout reward to
   every trained token through the group-relative advantage. NCCL weight sync
   keeps the vLLM server on the current policy, so the agent always samples
   from the model being trained.

Each rollout gets its own isolated session: one sandbox, one proxy port, one
agent process. Three small functions adapt the recipe to your task:
`rollout_reward_fn` (outcome to scalar reward), `train_turn_fn` (which turns
receive gradient), and `agent_turn_fn` (which trace entries are real agent
turns rather than auxiliary calls like title generation). All three are
documented in
[TRL's harness training guide](https://huggingface.co/docs/trl/openenv#training-on-harnesses-training-a-real-coding-agent-opencode).

## Run The Recipe

The reference script trains on competitive-coding problems from
`agentica-org/DeepCoder-Preview-Dataset`. The agent writes `solution.py`, and
the verifier runs it against held-out tests, returning the fraction passed.

```bash
pip install trl trackio datasets
pip install "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env"
```

Serve the policy with tool calling, token ids, and NCCL weight sync enabled
(one GPU), then train (a second GPU):

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B-Instruct-2507 \
    --host 0.0.0.0 --port 8000 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/opencode.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --vllm-url http://localhost:8000
```

The script is self-contained and runs the agent in a local subprocess sandbox,
so no container setup is needed. The recipe has been validated end to end on
Qwen3 (see [huggingface/trl#6420](https://github.com/huggingface/trl/pull/6420)).

## Full Recipe

- [Training on harnesses](https://huggingface.co/docs/trl/openenv#training-on-harnesses-training-a-real-coding-agent-opencode)
  in TRL's OpenEnv docs: rollout semantics, the reward path, turn selection,
  and the trace contract.
- [`examples/scripts/openenv/opencode.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/opencode.py)
  in TRL: the complete, runnable script.
- [`envs/opencode_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/opencode_env):
  the OpenEnv side, including the session factory, sandbox backends, and the
  transparent interception proxy.
