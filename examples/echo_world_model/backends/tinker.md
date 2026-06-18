# ECHO on Tinker

[Tinker](https://thinkingmachines.ai) (Thinking Machines) exposes four
primitives — `sample` / `forward_backward` / `optim_step` / `save_state` — and
lets *you* own the RL loop while the service owns the GPUs. ECHO fits its
`forward_backward` cleanly, because of one observation (also made by Prime
Intellect in *"True Agents Model the World"*):

> **The env-token loss is just SFT on the observation tokens, and SFT is RL with
> a constant positive advantage.** So you don't need a second loss function —
> you reuse the exact same `forward_backward` and only change the **per-token
> advantage vector**.

## The datum: one advantage per token

For each rollout you already build with [`trajectory.py`](../trajectory.py),
emit a per-token advantage:

```python
# action tokens  -> GRPO group-relative advantage A_i (can be negative)
# env_output tokens (obs_mask) -> a constant positive advantage = λ  (the world-model term)
# everything else (context, warnings) -> 0
advantages = torch.zeros(T)
advantages[action_mask] = group_relative_advantage          # standard GRPO
advantages[obs_mask]    = world_model_coeff                 # ECHO, as constant +adv SFT
```

Then the usual Tinker step trains both at once:

```python
datum = tinker.Datum(model_input=token_ids, loss_fn_inputs={"advantages": advantages, ...})
training_client.forward_backward([datum], loss_fn="importance_sampling")
training_client.optim_step(adam_params)
```

Notes (from the Prime Intellect write-up):
- Skip KL / importance-ratio / icepop masking *on the SFT (obs) tokens* — they
  are only needed for the RL (action) tokens.
- Normalize the RL and SFT token contributions **independently** so the dense
  env tokens don't drown out the sparse action tokens.
- Keep λ small — they saw collapse at 0.05 for GLM-4.5-Air, stable at 0.005;
  echo-rl's published Qwen3-8B config uses 0.05. Sweep it.
