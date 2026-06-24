# ECHO on TRL

[TRL](https://huggingface.co/docs/trl) is the framework OpenEnv recommends for RL
training (see `docs/source/guides/rl-integration.md` and the Wordle-GRPO
tutorial), so it is the natural place to run ECHO from an OpenEnv rollout.

Unlike [SkyRL](./skyrl.md), TRL does **not** ship an ECHO implementation today.
This note describes the pattern, the same way the [Tinker](./tinker.md) and
[Foundry Fine-Tuning](./foundry-finetuning.md) notes do. It is propositional.

## The masks map onto what TRL already has

| OpenEnv `trajectory.py` | TRL `GRPOTrainer` |
|---|---|
| `action_mask` | the completion / assistant-token mask (the GRPO target) |
| `obs_mask` (`env_only`) | the tool/env-response tokens TRL **masks out** of the loss today |
| `warning_mask` | harness boilerplate, excluded from the env loss |

The key asymmetry: TRL's multi-turn path keeps the tool-response tokens in the
sequence but **masks them out** of the policy-gradient loss (they were not
model-generated). That mask-out is exactly the "discard the environment" behavior
ECHO reverses. The `obs_mask` is the complement of the assistant mask within the
completion, derivable from the chat template's `{% generation %}` markers.

Producing that mask is OpenEnv's side (RFC 010 Part A): the multi-turn rollout via
the `environment_factory` integration must carry `obs_mask`, or you rebuild it by
hand as this example's [`trajectory.py`](../trajectory.py) does for the demo.

## The one change: a `GRPOTrainer` subclass

Override the loss to add a length-normalized cross-entropy on the observation
tokens, **reusing the per-token logps GRPO already computes** (recomputing them in
a second forward pass would forfeit ECHO's "~free" property):

```python
# illustrative — add λ·CE on the env-observation tokens, reusing GRPO's logps
class EchoGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, world_model_coeff=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.world_model_coeff = world_model_coeff

    def _compute_loss(self, model, inputs):
        # GRPO loss + the same per-token logps it already produced
        grpo_loss, per_token_logps = super()._compute_loss(model, inputs)
        obs_mask = inputs["obs_mask"]                      # from the OpenEnv rollout
        l_env = -(per_token_logps * obs_mask).sum() / obs_mask.sum().clamp(min=1.0)
        return grpo_loss + self.world_model_coeff * l_env
```

`world_model_coeff = 0.0` is exactly vanilla GRPO. The env term is the same
length-normalized cross-entropy as [`echo_loss.py`](../echo_loss.py).

## Notes (shared with the other backends)

- Apply the GRPO importance-ratio / clipping / KL **only** to the action tokens,
  not to the env (obs) cross-entropy term.
- Normalize the RL (action) and SFT (obs) contributions **independently** so the
  dense env tokens do not drown out the sparse action tokens.
- Keep λ small. Prime Intellect saw collapse at `0.05` for GLM-4.5-Air (stable at
  `0.005`); echo-rl's published Qwen3-8B config uses `0.05`. Expose it and sweep.
- Use `world_loss_target=env_only` and the rollout filters to keep harness
  boilerplate and pure-retrieval outputs out of the env term.

> For an open, reproducible run that already implements ECHO, use
> [SkyRL](./skyrl.md). This note is the path for the TRL-native stack.
