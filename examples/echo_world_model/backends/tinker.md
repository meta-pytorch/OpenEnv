# ECHO on Tinker

[Tinker](https://tinker-docs.thinkingmachines.ai/tinker/) exposes the training
primitives while its service owns the GPUs. The runnable example here is the
Tinker equivalent of [`train_echo.py`](../train_echo.py): verifier-free
world-model training on the same OpenEnv-shaped terminal rollouts and the same
held-out env-token cross-entropy metric.

## Run it

From `examples/echo_world_model`:

```bash
export TINKER_API_KEY="..."
uv run backends/tinker_echo_demo.py --steps 15
```

The script's inline dependencies pin `tinker==0.22.7`; `uv run` installs them.
Tinker executes remote training and requires an account, API key, and available
credits. The default model is `Qwen/Qwen3.5-4B` with a rank-16 LoRA. Override it
with `--model`, `--rank`, and `--learning-rate`. Add
`--checkpoint-name echo-world-model` to persist weights and optimizer state.

## How the role masks map to Tinker

For each rollout from [`trajectory.py`](../trajectory.py), the example creates a
causal-language-model `Datum`:

```python
datum = tinker.types.Datum(
    model_input=tinker.types.ModelInput.from_ints(token_ids[:-1]),
    loss_fn_inputs={
        "target_tokens": token_ids[1:],
        "weights": normalized_obs_mask[1:],
    },
)
```

The shift matters: input position `t` predicts target token `t + 1`, so the loss
weight comes from the **target** token's role. Only `env_output` has non-zero
weight; action, context, and warning tokens remain zero-weight conditioning
context. Batch-wide normalization makes Tinker's sum-reduced cross-entropy equal
the mean env-token CE used by the local demo.

The training step follows Tinker's documented SFT path:

```python
fwdbwd = training_client.forward_backward(data, "cross_entropy")
optim = training_client.optim_step(tinker.types.AdamParams(learning_rate=2e-4))
result = fwdbwd.result()
optim.result()
```

`TrainingClient.forward(..., "cross_entropy")` evaluates the train and held-out
data without accumulating gradients, so the printed metric is comparable to the
local example.

## Scope

This script intentionally matches the existing **verifier-free** demo
(`use_rl=False`): it proves that environment responses alone train a world
model. It does not claim to implement the full `L_GRPO + lambda * L_env`
objective. A full Tinker RL adapter must keep importance ratios/clipping on
action tokens only, keep plain CE on env tokens, and normalize the two
contributions independently—use a single custom loss if preserving ECHO's
one-forward-pass property.
