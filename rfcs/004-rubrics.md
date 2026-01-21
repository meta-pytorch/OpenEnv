# RFC 004: Rubric System for Reward Computation

## Summary

This RFC introduces the Rubric system—a composable, nn.Module-inspired abstraction for computing rewards in OpenEnv environments. Environment authors implement `__init__` and `forward(action, observation) -> float`; the framework handles composition, observability, and parallel evaluation.

---

## Motivation

**Problem 1: No standard reward interface**

Currently, environment authors must roll their own reward computation. This leads to:
- Inconsistent APIs across environments
- No reusability of common patterns (gating, weighted sums)
- Training frameworks can't introspect reward components

**Problem 2: Multi-criteria evaluation is hard**

Modern RL requires evaluating outputs on multiple dimensions (correctness, style, safety). Without a composition abstraction:
- Authors manually aggregate scores with ad-hoc logic
- No standard patterns for hierarchical gating ("must compile before style matters")
- No visibility into which criteria caused low rewards

**Problem 3: Parallel evaluation for I/O-bound rubrics**

LLM judges and sandboxed execution are slow (100-1000ms). Without async support:
- Batch evaluation becomes a bottleneck
- Training throughput suffers

---

## Proposed Solution

A **Rubric** is a callable that computes a reward:

```python
class Rubric:
    def forward(self, action, observation) -> float:
        """Implement this. Return 0.0-1.0."""
        raise NotImplementedError
```

The API is modeled after PyTorch's `nn.Module`:
- Users implement `__init__` and `forward`
- Child rubrics auto-register when assigned as attributes
- Hooks provide observability without polluting the base class
- `state_dict()` / `load_state_dict()` for serialization

### Rubrics Live Inside Environments

Rubrics are **server-side only**. Each environment defines its rubric in `__init__`, and the rubric executes during `step()`. Access via `env.rubric`:

```python
class CodeEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.rubric = CodeRubric()  # Required attribute

    def step(self, action: CodeAction) -> CodeObservation:
        # ... execute action ...
        reward = self.rubric(action, observation)
        return observation.with_reward(reward)
```

Training infrastructure accesses rubrics for introspection:

```python
# Log all component scores
for name, r in env.rubric.named_rubrics():
    print(f"{name}: {r.last_score}")
```

**Important**: The `Environment` base class requires a `rubric` attribute. All environments must define `self.rubric` in their constructor.

---

## Usage Examples

### Basic Rubric

A leaf rubric implements a single evaluation criterion:

```python
class Compiles(Rubric):
    def forward(self, action, observation) -> float:
        try:
            ast.parse(action.content)
            return 1.0
        except SyntaxError:
            return 0.0
```

### Composite Rubrics

The nn.Module-like design enables hierarchical composition, a pattern we see across recent academic work on rubric-based rewards. Rubicon (Huang et al., 2024) uses multi-dimensional rubrics with gating; AdvancedIF (He et al., 2024) combines multiple criteria with all-or-nothing aggregation; RLTF (Liu et al., 2023) applies hierarchical evaluation with fail-fast semantics.

In our design, child rubrics are automatically registered when assigned as attributes, enabling patterns like:

```python
class CodeRubric(Rubric):
    def __init__(self, test_weight: float = 0.7):
        super().__init__()
        self.test_weight = test_weight
        self.compiles = Compiles()      # Auto-registered
        self.tests = PassesTests()
        self.style = LLMStyleJudge()

    def forward(self, action, observation) -> float:
        if self.compiles(action, observation) < 1.0:
            return 0.0  # Gate: must compile
        t = self.tests(action, observation)
        s = self.style(action, observation)
        return t * self.test_weight + s * (1 - self.test_weight)
```

The framework can traverse the rubric hierarchy via `named_rubrics()`, enabling automatic logging of all component scores.

### Container Rubrics

The core Rubric spec is just `forward(action, observation) -> float`. Together with the spec, we provide container rubrics for common aggregation patterns, similar to how PyTorch provides `nn.Sequential` alongside `nn.Module`:

- **Sequential**: Runs child rubrics in order. If any returns 0, stops immediately and returns 0 (fail-fast). This implements the hierarchical gating pattern from RLTF where syntax checks run before execution checks.

- **Gate**: Wraps a rubric with a threshold. Returns the child's score if it meets the threshold, otherwise returns 0. Useful for hard constraints ("must pass 50% of tests").

- **WeightedSum**: Computes a weighted combination of child rubrics. The standard aggregation pattern, though research (Gonzalez et al., 2024) suggests holistic judging sometimes outperforms point-summing.

- **RubricList**: A container for dynamic lists of rubrics, analogous to `nn.ModuleList`. Does not define aggregation—use within a parent rubric that implements custom logic.

- **RubricDict**: A container for named rubrics, analogous to `nn.ModuleDict`. Enables keyed access for multi-task environments where different tasks require different rubrics:

  ```python
  class AtariRubric(Rubric):
      def __init__(self):
          super().__init__()
          self.games = RubricDict({
              "pong": PongRubric(),
              "breakout": BreakoutRubric(),
              "space_invaders": SpaceInvadersRubric(),
          })

      def forward(self, action, obs) -> float:
          return self.games[obs.game_id](action, obs)
  ```

  Access: `env.rubric.games["pong"]`

- **LLMJudge**: A rubric that calls an LLM endpoint to evaluate the response. Takes a prompt template and endpoint configuration. The LLM call happens via the configured MCP service, keeping the rubric portable across deployments.

These containers compose naturally:

```python
# Hierarchical gating: compile → pass tests → score style
rubric = Sequential(
    Gate(Compiles()),
    Gate(PassesTests(), threshold=0.5),
    WeightedSum([PassesTests(), StyleRubric()], weights=[0.7, 0.3])
)
```

### Observability via Hooks

Just like `nn.Module`, we support pre- and post-forward hooks. Hooks are called before and after `forward()` executes, without modifying the rubric's return value. This is particularly useful for observability—training systems can install hooks to capture every component score for logging and debugging, without rubric authors needing to do anything special.

```python
scores = {}
for name, r in rubric.named_rubrics():
    r.register_forward_hook(
        lambda r, a, o, result, n=name: scores.__setitem__(n, result)
    )

reward = rubric(action, observation)
wandb.log(scores)  # {"compiles": 1.0, "tests": 0.8, "style": 0.7}
```

Hooks also enable extensions like caching, timing, or validation without modifying the base class.

### Batch Evaluation via Environment Stacking

**Key architectural decision**: Environments do not support multiplexed trajectories. One environment instance handles one trajectory. To generate batches, stack multiple environment instances.

Rubric authors write synchronous `forward()` methods—no async knowledge required. For batch evaluation, the training infrastructure manages a pool of environments:

```python
async def train():
    # Stack of 64 environments, each with its own rubric
    envs = EnvPool("code_env", n=64)

    for batch in dataloader:
        actions = await model.generate(batch.prompts)
        # EnvPool orchestrates calls across all environments
        observations = await envs.step_batch(actions)
        rewards = [obs.reward for obs in observations]
        loss = compute_loss(rewards)
```

Each environment computes its own reward via `self.rubric(action, observation)` during `step()`. The `EnvPool` helper orchestrates parallel execution across the stack—training code never instantiates rubrics directly.

For single-environment scripts, just call `env.step(action)` synchronously. The reward is computed inside the environment and returned in the observation.

---

## What Changes

| Component | Change |
|-----------|--------|
| New `Rubric` base class | `forward(action, observation) -> float`, auto-registration, hooks, `get_rubric()` |
| New containers | `Sequential`, `Gate`, `WeightedSum`, `RubricList`, `RubricDict`, `LLMJudge` |
| `Environment` base class | Requires `rubric: Rubric` attribute; `step()` uses rubric for reward |
| New `EnvPool` helper | Orchestrates batch evaluation across stacked environments |

### Environment Base Class Changes

The `Environment` base class gains a required `rubric` attribute:

```python
class Environment(Generic[ActT, ObsT, StateT]):
    rubric: Rubric  # Required - must be set in __init__

    def step(self, action: ActT) -> ObsT:
        # Implementation calls self.rubric(action, observation)
        # Reward returned in observation
        ...
```

All environments must define `self.rubric` in their constructor. The framework validates this at environment startup.

---

## API Reference

### Rubric Base Class

| Method | Description |
|--------|-------------|
| `forward(action, observation) -> float` | Implement this. Compute reward. |
| `__call__(action, observation) -> float` | Sync evaluation with hooks. |
| `evaluate(action, observation) -> float` | Async evaluation via thread pool. |
| `register_forward_hook(fn)` | Called after `forward()`. Signature: `(rubric, action, obs, result)` |
| `register_forward_pre_hook(fn)` | Called before `forward()`. Signature: `(rubric, action, obs)` |
| `children()` / `named_children()` | Iterate immediate child rubrics. |
| `rubrics()` / `named_rubrics()` | Iterate all descendant rubrics. |
| `get_rubric(path: str)` | Access nested rubric by dot-separated path (e.g., `"code.syntax"`). |
| `state_dict()` / `load_state_dict(d)` | Serialize/deserialize configuration. |

### Container Rubrics

| Container | Behavior |
|-----------|----------|
| `Sequential(*rubrics)` | Run in order; stop and return 0 if any returns 0 |
| `Gate(rubric, threshold=1.0)` | Return 0 if score < threshold |
| `WeightedSum(rubrics, weights)` | Weighted combination |
| `RubricList(rubrics)` | Container for dynamic lists (no aggregation) |
| `RubricDict(rubrics: Dict[str, Rubric])` | Container for named rubrics with keyed access |
| `LLMJudge(prompt_template, endpoint)` | Call LLM via MCP for evaluation |

---

## Alternatives Considered

**Two separate classes (Criterion + Rubric)**

We considered having `Criterion` for leaf evaluators and `Rubric` for composites. This mirrors some academic terminology. However, the distinction adds complexity without benefit—a rubric with no children is just a leaf. PyTorch's `nn.Module` doesn't distinguish between "layer" and "model"; neither do we.

**Declarative YAML/JSON schema**

A declarative format would enable non-programmers to define rubrics and allow static validation. However, interesting rubrics involve LLM calls, sandboxed execution, database queries, and complex branching logic. These don't fit declarative formats well. Python code is more expressive and debuggable. The ecosystem (TRL, OpenRLHF, veRL) has converged on code-based rewards.

**Async forward() everywhere**

We considered making `forward()` async by default, which would enable natural parallelism within a single rubric. However, this forces all rubric authors to understand async/await, even for trivial synchronous checks like `ast.parse()`. Our approach—sync `forward()` with async `evaluate()` wrapper—keeps rubric authoring simple while enabling batch-level parallelism in training loops.

**Magic proxy types for automatic parallelism**

We prototyped a design where `rubric()` returns a proxy object that resolves lazily, enabling automatic parallelism like PyTorch's CUDA tensors. This proved fragile: overriding `__add__`, `__lt__`, and other operators led to confusing edge cases. Explicit `evaluate_batch()` is clearer and sufficient for the primary use case (batch evaluation in training).

---

## Open Questions

**How do rubrics access external services (LLM endpoints, databases)?**

LLM judges and other rubrics need to call external services. For now, rubrics can simply make RPC calls inside their `forward()` as needed—there are no restrictions on what a rubric can do internally. A future RFC will address reusable patterns for remote resources (connection pooling, retry logic) and lifecycle management of those connections.

**Error handling in forward()**

Currently, exceptions in `forward()` propagate to the caller. Rubric authors should handle expected errors (network timeouts, invalid inputs) internally. We may add optional error handling in the base class (e.g., return 0.0 on exception with logging) based on user feedback.

**Thread pool sizing**

The `evaluate_batch()` helper accepts an optional `max_workers` parameter to configure thread pool size. The default is 32 workers.

---

## Implementation Plan

This RFC will be implemented in stacked PRs:

### PR 1: Rubric Base Class

- `Rubric` base class with `forward()`, `__call__()`, child auto-registration
- Container rubrics: `Sequential`, `Gate`, `WeightedSum`, `RubricList`
- `RubricDict` for multi-task dispatch
- `get_rubric(path)` for nested access
- Unit tests for all containers

### PR 2: Environment Integration

- Update `Environment` base class to require `rubric` attribute
- Update `step()` to wire rubric output to observation reward
- Update environment documentation

### PR 3: Migrate Existing Environments

- Migrate `textarena_env` from `RewardProvider` to `Rubric`
- Update any other environments using custom reward patterns

### PR 4: EnvPool (Future)

- Batch orchestration across stacked environments
- `step_batch()` helper for parallel execution
- May require separate RFC depending on scope

---

## Appendix: Patterns from Literature

The design supports patterns from recent academic work on rubric-based rewards:

| Pattern | Source | Implementation |
|---------|--------|----------------|
| Hierarchical gating | Liu et al., "RLTF: Reinforcement Learning from Unit Test Feedback" (TMLR 2023) | `Sequential(Gate(...), Gate(...), ...)` |
| All-or-nothing rewards | He et al., "AdvancedIF: Rubric-Based Instruction Following" (2024) | Custom `forward()` returning 1.0 only if all criteria pass |
| Multi-dimensional rubrics | Huang et al., "Rubicon: Reinforcement Learning with Rubric Anchors" (2024) | Composite rubrics with multiple children |
| Defense rubrics | Huang et al. (2024), He et al. (2024) | Gate with anti-hacking rubrics (e.g., `NoSycophancy`) |
| Gatekeeper mechanism | Liu et al., "OpenRubrics: Scalable Synthetic Rubric Generation" (2024) | `Sequential(hard_rules..., WeightedSum(soft_principles...))` |
| Implicit aggregation | Gonzalez et al., "Rubrics as Rewards" (Scale AI, 2024) | `LLMJudge` with all criteria in prompt for holistic scoring |
