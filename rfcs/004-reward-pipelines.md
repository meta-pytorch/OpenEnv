# RFC 004: Reward Pipelines

| Field | Value |
|-------|-------|
| Status | Draft |
| Author | OpenEnv Team |
| Created | 2025-12-05 |
| Depends on | RFC 003 (MCP Support) |

## Summary

This RFC standardizes how environments compute rewards. We introduce **rubrics** - composable reward functions that environments call internally to score agent behavior. Rubrics can range from simple Python checks (regex, range validation) to LLM judges that require external RPC. For external calls, rubrics use the MCP client established in RFC 003, keeping environment code portable across deployments.

Design philosophy: **minimal spec + strong SDK**.

## Motivation

As RL becomes mainstream, reward engineering becomes a primary lever for improving agent capabilities. Many engineers will contribute mainly to reward functions rather than environment logic. OpenEnv needs to support:

- **Composition**: Combine accuracy, style, and safety rubrics with configurable weights
- **Dynamic schedules**: Decay reward shaping over training (like AlphaStar's approach to curriculum learning)
- **Cross-environment normalization**: When Alice's env returns 0-100 and Bob's returns 0-1, training needs comparable scales
- **Observability**: Per-rubric scores in W&B for debugging what's driving learning

## What's Changing

| Current | Proposed | Impact |
|---------|----------|--------|
| Rewards computed ad-hoc in `step()` | Standardized `Rubric` protocol | Reusable, composable |
| No standard structure | `metadata["reward_components"]` | Per-rubric logging |
| Config immutable | `POST /config` endpoint | Dynamic weight schedules |
| No helpers | SDK: `RubricComposer`, `RewardNormalizer` | Best practices by default |

## Rubrics

A rubric is a callable that scores agent behavior. The environment calls rubrics internally during `step()` - they are never exposed to the agent.

```python
class Rubric(Protocol):
    def __call__(self, action: Action, observation: Observation) -> float:
        """Return score in [0, 1]."""
        ...
```

### Pure Python Rubrics

Simple checks run locally with no external dependencies:

```python
class FormatRubric:
    def __call__(self, action, observation) -> float:
        return 1.0 if action.response.startswith("Answer:") else 0.0
```

### Rubrics with External Services

When a rubric needs an LLM judge or database, it uses the environment's `mcp_client` from RFC 003. This decouples environment code from specific endpoints - training infra provides the mapping at deployment time.

```python
class StyleRubric:
    def __init__(self, config, mcp_client: MCPClient):
        self.config = config
        self.mcp = mcp_client

    def __call__(self, action, observation) -> float:
        result = self.mcp.call_tool(
            self.config.judge_service,  # e.g., "style_judge"
            text=action.message
        )
        return result["score"]
```

The same environment works on Alice's cluster (where `style_judge` points to her VLLM instance) and Bob's laptop (where it points to Ollama). Environment authors write to service names; infrastructure owners map those names to real endpoints.

## Observation Format

Environments return a scalar reward plus per-rubric breakdown in metadata:

```python
Observation(
    reward=0.85,
    metadata={
        "reward_components": {
            "accuracy": 0.9,
            "style": 0.7
        }
    }
)
```

This enables per-rubric curves in experiment tracking while keeping the core API simple.

## Configuration

Environments define which rubrics to run and when:

```yaml
schema_version: "1.0"

per_turn:
  - name: toxicity
    rubric: myenv.ToxicityRubric
    weight: -10.0

episode_end:
  - name: task_success
    rubric: myenv.AccuracyRubric
    weight: 1.0
```

Config is passed at construction:

```python
client = HTTPEnvClient.from_docker_image("coding_env:v1.0", config=cfg)
```

### Dynamic Updates

Training can adjust weights mid-run for reward shaping schedules:

```python
for episode in range(10000):
    if episode % 1000 == 0:
        weight = 1.0 - (episode / 10000)  # Decay shaping over time
        env.post("/config", json={"per_turn": [{"name": "shaping", "weight": weight}]})
```

The `POST /config` endpoint accepts partial updates. When updates take effect is implementation-defined; convention is at the next `reset()`.

### Schema Versioning

If config includes `schema_version` and it doesn't match the environment's expected version, the environment rejects it with a clear error. If version is omitted, the environment accepts the config permissively and fills missing fields with defaults.

## SDK Helpers

The SDK provides optional helpers that encode best practices.

**RubricComposer** handles within-environment composition with automatic normalization:

```python
composer = RubricComposer(
    rubrics={"accuracy": AccuracyRubric(), "style": StyleRubric(cfg, mcp)},
    weights={"accuracy": 1.0, "style": 0.5},
    normalize=True
)
reward, components = composer.compute(action, obs)
```

**RewardNormalizer** handles cross-environment normalization on the training side:

```python
normalizer = RewardNormalizer(method="running_mean_std", per_environment=True)
normalized = normalizer(obs.reward, env_id=env.id)
```

| Method | Use Case |
|--------|----------|
| `running_mean_std` | General purpose (PPO-style) |
| `min_max` | Unknown bounds |
| `percentile` | Sparse/noisy rewards |

## Ownership

The environment owns rubric code and config schema. Training infrastructure owns config values, weight schedules, MCP service endpoints, and cross-environment normalization.

| Aspect | Owner |
|--------|-------|
| Rubric code | Environment |
| Config schema | Environment |
| Config values, weights, schedules | Training infra |
| MCP service endpoints | Training infra |
| Cross-env normalization | Training infra |

## Multi-Turn Scenarios

For agentic evals like Tau-bench, the environment orchestrates multiple roles: a Service Model (simulated user) and a Judge. Both are accessed via `mcp_client` as external services. The environment manages conversation state internally; external services are stateless, receiving full conversation history each call.

## Conventions

| Convention | Rationale |
|------------|-----------|
| Rubrics return [0, 1] | Composable without rescaling |
| Include `reward_components` in metadata | Enables per-rubric logging |
| Use MCP for external services | Portable across deployments |
| Accept `POST /config` for weights | Enables reward shaping |

## FAQ

**Q: Are rubrics exposed to the agent?**
No. Rubrics are internal to the environment. The agent interacts via MCP tools (RFC 003); rubrics compute rewards after the agent acts.

**Q: How do I call an LLM judge?**
Use `mcp_client.call_tool()` inside your rubric. Training infra maps service names to actual endpoints.

**Q: What if my rubric needs trajectory history?**
Access it via `observation.metadata`. The environment maintains state; rubrics themselves are stateless.

## References

- RFC 003: MCP Support
