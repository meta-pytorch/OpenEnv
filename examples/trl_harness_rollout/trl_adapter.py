"""TRL integration skeleton (this is the part to implement on the trainer side).

This file is a STUB, not a working integration. It marks exactly where TRL plugs into the OpenEnv
harness rollout worker. Nothing here imports TRL, so the rest of this draft runs without TRL installed.

The OpenEnv side (rollout_worker.py + interception.py + harness.py) is done and runnable. To turn this
draft into real online RL training, implement the two seams below.

----------------------------------------------------------------------------------------------------
SEAM 1: generation + token capture + tokenization (TITO)
----------------------------------------------------------------------------------------------------
Implement `GenerateAPI` backed by the trainer's vLLM. `generate.VLLMGenerate` is a working starting
point, BUT its tokenization is naive. The real version owns:
  - generate from the current policy (on-policy), capturing exact token_ids + logprobs per turn
  - messages -> tokens with a prefix-preserving, token-in/token-out scheme (TITO): never re-encode
    decoded tokens, only encode the new interstitial between turns, keep an exact completion mask.

    class TrlGenerate:  # satisfies rollout_worker.GenerateAPI
        def generate(self, *, rollout_id, turn, messages, tools, sampling) -> str:
            # TODO: render messages -> prompt_ids (TITO-safe), generate with the trainer's vLLM,
            #       store (prompt_ids, completion_ids, logprobs) keyed by (rollout_id, turn).
            ...

----------------------------------------------------------------------------------------------------
SEAM 2: consume the message-level rollout into a training sample
----------------------------------------------------------------------------------------------------
The worker emits `RolloutMessages{rollout_id, messages, reward}`. The trainer stitches the per-turn
tokens it captured in Seam 1 into one flat (input_ids, completion_mask, old_log_probs), applies the
reward, computes the group-relative advantage, and feeds the loss.

    Open question: since Seam 1 already captured exact ids per (rollout_id, turn), the trainer may only
    need {rollout_id, reward} from Seam 2 and can reconstruct the rest from its captures. The worker
    emits the full transcript to be safe.

----------------------------------------------------------------------------------------------------
WIRING into AsyncGRPO (a RolloutWorkerProtocol adapter around the OpenEnv worker)
----------------------------------------------------------------------------------------------------
TRL's AsyncGRPOTrainer accepts a custom `rollout_worker` implementing RolloutWorkerProtocol
(a `rollout_buffer` queue + start/stop/update_model_version/check_health). Wrap the OpenEnv worker:

    import queue
    from openenv.core.harness.rollout_worker import HarnessRolloutWorker

    class TrlRolloutWorkerAdapter:           # satisfies trl ... RolloutWorkerProtocol
        def __init__(self, openenv_worker: HarnessRolloutWorker, tasks):
            self.rollout_buffer = queue.Queue()
            self._worker = openenv_worker
            self._tasks = tasks
            self._model_version = 0
            # TODO: thread pool / inflight control; pull tasks, run worker.produce, then for each
            #       RolloutMessages build the training sample via Seam 1 captures + TITO and push it.
        def start(self): ...   # begin producing into rollout_buffer
        def stop(self): ...
        def update_model_version(self, v): self._model_version = v
        def check_health(self, stale_after_s): ...

    # trainer = AsyncGRPOTrainer(
    #     model=..., args=AsyncGRPOConfig(vllm_server_base_url=..., num_generations=...),
    #     reward_funcs=lambda **kw: [0.0],          # unused: reward comes from the env's verify()
    #     rollout_worker=TrlRolloutWorkerAdapter(openenv_worker, tasks),
    # )
    # trainer.train()

----------------------------------------------------------------------------------------------------
WEIGHT SYNC CAVEAT (must be resolved on the trainer side for on-policy training)
----------------------------------------------------------------------------------------------------
In TRL today, injecting a custom `rollout_worker` sets `self.weight_transfer = None`
(async_grpo_trainer.py, the `if rollout_worker is not None:` branch), and `_sync_weight()` only acts
`if self.weight_transfer:`. It does NOT call `rollout_worker.send_weights(...)`. So with an injected
worker, the trainer does NOT push updated weights to vLLM by default, and generation can drift
off-policy.

The trainer side must handle this, by one of:
  - implement weight sync inside the worker (a `send_weights` path the worker drives), or
  - extend `_sync_weight()` to also call `rollout_worker.send_weights(...)` when `weight_transfer` is
    None (a small TRL change).
This is a trainer-side concern, flagged here so it is not missed. It does not affect the OpenEnv worker.

----------------------------------------------------------------------------------------------------
DIVISION (what is done vs to do)
----------------------------------------------------------------------------------------------------
Done (OpenEnv side, this draft):
  - HarnessRolloutWorker: drives the harness via interception, verify, emits message-level rollouts
  - interception proxy (stand-in for OpenEnv InterceptionServer, PR #694)
  - a harness with the right dynamic (agent owns its loop, multi-turn)
  - runnable with no GPU (fake) and with real vLLM (token capture)

To do (trainer side):
  - TrlGenerate (Seam 1): on-policy generation + capture + TITO
  - TrlRolloutWorkerAdapter (Seam 2 + wiring): message-level rollout -> training sample -> AsyncGRPO
  - Weight sync to vLLM (see the caveat above): not handled by default with an injected worker
"""
