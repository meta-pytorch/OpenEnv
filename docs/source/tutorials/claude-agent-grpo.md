# Coding Agent Training with TRL (Claude Code)

This tutorial covers the black-box training path: training the actual
[Claude Code](https://github.com/anthropics/claude-code) coding agent, with its
own planner, tools, context management, and stop condition, using TRL's
experimental `AsyncGRPOTrainer`. The agent owns its loop, and OpenEnv captures
what it did.

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
   `ClaudeCodeSessionFactory` from
   [`claude_code_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/claude_code_env),
   in `transparent_proxy` mode. A small proxy inside the sandbox forwards the
   agent's `/v1/chat/completions` calls to your vLLM server and records each
   turn's token ids and logprobs to a trace. Claude Code speaks the Anthropic
   Messages API, so an in-sandbox Anthropic-to-OpenAI shim sits in front of the
   interception proxy and translates each request, leaving the capture point
   unchanged.
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
turns rather than auxiliary calls like title generation).

## Full Recipe

The reference script trains on competitive-coding problems from
`agentica-org/DeepCoder-Preview-Dataset`: the agent writes `solution.py`, and
the verifier runs it against held-out tests, returning the fraction passed. It
is self-contained, runs the agent in a local subprocess sandbox (no container
setup needed), needs two GPUs (one serving the policy with vLLM, one
training).

- [`examples/scripts/openenv/claude.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/openenv/claude.py)
  in TRL: the complete, runnable script.
- [`envs/claude_code_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/claude_code_env):
  the OpenEnv side, including the session factory, sandbox backends, and the
  transparent interception proxy.
