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
    from rollout_worker import HarnessRolloutWorker

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
WEIGHT SYNC (trainer-side, only on the injected-worker path)
----------------------------------------------------------------------------------------------------
Weight sync is a trainer-side concern. TRL already handles it on the DEFAULT path
(`environment_factory` + the built-in `AsyncRolloutWorker`): it builds a `WeightTransferClient` and
`_sync_weight()` pushes weights to vLLM.

The boundary to know: when you INJECT a custom `rollout_worker`, TRL sets `self.weight_transfer = None`
(async_grpo_trainer.py, the `if rollout_worker is not None:` branch), and `_sync_weight()` only acts
`if self.weight_transfer:`, so it becomes a no-op. It also calls `weight_transfer.send_weights(...)`,
never `rollout_worker.send_weights(...)`. So on the injected-worker path the trainer side must wire
weight sync itself, by one of:
  - worker-driven sync (the worker owns `send_weights` + `update_model_version`, the way OpenEnv PR
    #695's worker does), or
  - extending `_sync_weight()` to also call `rollout_worker.send_weights(...)` (a small TRL change).
Not a bug, the known boundary of the injected-worker path. It does not affect the OpenEnv worker.

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
