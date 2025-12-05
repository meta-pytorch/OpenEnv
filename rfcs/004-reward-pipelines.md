# RFC 004: Reward Pipelines

**Status:** Draft
**Author:** OpenEnv Team
**Created:** 2025-01-XX
**Last Updated:** 2025-01-XX

## Summary

This RFC defines how reward functions (rubrics) are structured, composed, configured, and integrated with training infrastructure in OpenEnv. The design follows a "minimal spec + strong SDK" philosophy: the specification defines only essential contracts while the SDK provides well-designed helpers that guide users toward best practices.

## Motivation

Reward engineering is becoming a primary lever for training capable AI agents. As RL becomes mainstream, many engineers will contribute primarily or exclusively to reward rubrics. OpenEnv must support:

1. **Reusability** - Rubrics that can be shared and built upon
2. **Composition** - Multiple rubrics combined into a single reward signal
3. **Versioning** - Track reward function changes for reproducibility
4. **Dynamic configuration** - Adjust reward weights during training (e.g., reward shaping schedules)
5. **Cross-environment normalization** - Comparable rewards across different environments

### Design Principles

Following PyTorch's philosophy:

1. **Minimal spec** - Stay flexible, don't over-constrain
2. **Strong conventions** - Provide excellent defaults so the path of least resistance is correct
3. **SDK as gravity** - Optional helpers that most users adopt because they're well-designed

## Terminology

- **Rubric**: A reward function that scores agent behavior. Can be rule-based (Python function) or model-based (LLM judge).
- **Reward Pipeline**: The composition of multiple rubrics into a final scalar reward.
- **Reward Shaping**: Auxiliary rewards that guide learning but may be reduced/eliminated over training.

## Specification

### Observation Format

Observations include a scalar reward and flexible metadata:

```python
@dataclass
class Observation:
    done: bool
    reward: float              # Final weighted reward (scalar)
    metadata: dict             # Flexible, environment-defined
    # ... other fields
```

**Convention:** Environments with multiple rubrics SHOULD include `reward_components` in metadata:

```python
metadata = {
    "reward_components": {
        "accuracy": 0.85,
        "style": 0.72,
        "safety": 1.0
    },
    # ... other metadata
}
```

### Rubric Interface

Rubrics are callable classes with configuration:

```python
from typing import Protocol
from dataclasses import dataclass

class Rubric(Protocol):
    """Protocol for reward rubrics."""

    def __call__(self, action: Action, observation: Observation) -> float:
        """
        Compute reward score for the given action and observation.

        Args:
            action: The action taken by the agent
            observation: The resulting observation (includes metadata for context)

        Returns:
            Reward score. Convention: normalized to [0, 1] or [-1, 1].
        """
        ...
```

**Rubric Configuration:**

Each rubric has an associated config dataclass:

```python
@dataclass
class StyleRubricConfig:
    """Configuration for style rubric."""
    weight: float = 1.0
    llm_service: str = "style_judge"  # MCP service name
    temperature: float = 0.0

@dataclass
class AccuracyRubricConfig:
    """Configuration for accuracy rubric."""
    weight: float = 1.0
    timeout_seconds: float = 5.0
    partial_credit: bool = True
```

**Rubric Implementation:**

```python
class StyleRubric:
    """Rubric that uses an LLM to judge response style."""

    def __init__(self, config: StyleRubricConfig, mcp_client: MCPClient):
        self.config = config
        self.mcp = mcp_client

    def __call__(self, action: Action, observation: Observation) -> float:
        # Access context from observation metadata
        conversation = observation.metadata.get("conversation_history", [])

        # Call LLM judge via MCP
        result = self.mcp.call_tool(
            self.config.llm_service,
            text=action.message,
            context=conversation
        )

        return result["score"]  # Assumed to be in [0, 1]
```

### Rubric Types

**1. Pure Python Rubrics (Deterministic)**

Fast, rule-based checks:

```python
class RegexRubric:
    """Check if response matches expected pattern."""

    def __init__(self, config: RegexRubricConfig):
        self.pattern = re.compile(config.pattern)

    def __call__(self, action: Action, observation: Observation) -> float:
        if self.pattern.match(action.content):
            return 1.0
        return 0.0
```

**2. LLM Judge Rubrics (Via MCP)**

Complex, nuanced judgments:

```python
class NeutralToneRubric:
    """Check if response maintains neutral tone on sensitive topics."""

    def __init__(self, config: NeutralToneConfig, mcp_client: MCPClient):
        self.config = config
        self.mcp = mcp_client

    def __call__(self, action: Action, observation: Observation) -> float:
        result = self.mcp.call_tool(
            "neutral_tone_judge",
            text=action.message,
            rubric=self.config.rubric_prompt
        )
        return result["score"]
```

### Environment Configuration

Environments define their reward pipeline in configuration:

```python
@dataclass
class RewardConfig:
    """Reward pipeline configuration."""
    schema_version: str = "1.0"

    # Per-turn rubrics (evaluated every step)
    per_turn: List[RubricEntry] = field(default_factory=list)

    # Episode-end rubrics (evaluated when done=True)
    episode_end: List[RubricEntry] = field(default_factory=list)

@dataclass
class RubricEntry:
    """Single rubric in the pipeline."""
    name: str                    # Identifier for logging
    rubric: str                  # Rubric class path or name
    weight: float = 1.0          # Composition weight
    config: dict = field(default_factory=dict)  # Rubric-specific config
```

**Example YAML configuration:**

```yaml
schema_version: "1.0"

per_turn:
  - name: toxicity_check
    rubric: openenv.rubrics.ToxicityRubric
    weight: -10.0  # Heavy penalty
    config:
      threshold: 0.8

episode_end:
  - name: task_completion
    rubric: coding_env.rubrics.TestPassRubric
    weight: 1.0
    config:
      partial_credit: true

  - name: code_style
    rubric: coding_env.rubrics.StyleRubric
    weight: 0.3
    config:
      llm_service: style_judge
```

### Configuration Lifecycle

**Initial Configuration:**

Config is passed at environment construction:

```python
client = HTTPEnvClient.from_docker_image(
    "coding_env:v1.0",
    config=reward_config
)
```

The environment initializes rubrics and MCP connections based on this config.

**Dynamic Configuration Updates:**

Training infrastructure can update configuration during training:

```
POST /config
Content-Type: application/json

{
    "per_turn": [
        {"name": "reward_shaping", "weight": 0.5}  # Reduce shaping weight
    ]
}
```

**Response:**
```json
{
    "status": "ok",
    "config": { /* updated config */ }
}
```

**Implementation-defined behavior:**
- What fields are updateable
- When updates take effect (immediately vs. next reset)
- Validation rules

**Convention:** Most environments accept weight updates and apply them at the next `reset()`.

**Reading Current Configuration:**

```
GET /config

Response:
{
    "schema_version": "1.0",
    "per_turn": [...],
    "episode_end": [...]
}
```

### Schema Versioning

Config includes a schema version for compatibility checking:

```python
@dataclass
class RewardConfig:
    schema_version: str = "1.0"
    # ...
```

**Version Checking Behavior:**

| Config has version? | Environment behavior |
|---------------------|---------------------|
| Yes, matches | Accept config |
| Yes, mismatch | Reject with error |
| No | Accept, use defaults for missing fields, log warning |

This allows explicit version pinning for reproducibility while remaining permissive for quick iteration.

### External Services via MCP

All external services (LLM judges, databases, APIs) are accessed via MCP:

```python
class Environment:
    def __init__(self, config: EnvironmentConfig):
        # Initialize MCP client for external services
        self.mcp = MCPClient(config.mcp_endpoint)

        # Rubrics receive MCP client
        self.rubrics = [
            StyleRubric(config.style, self.mcp),
            SafetyRubric(config.safety, self.mcp),
        ]
```

**MCP Service Configuration:**

Training infrastructure provides MCP service endpoints:

```yaml
mcp_services:
  style_judge:
    type: openai_llm
    endpoint: http://vllm-service:8000
    model: llama-3.1-70b-instruct

  safety_check:
    type: openai_llm
    endpoint: http://safety-model:8000
    model: llama-guard-3
```

The environment code is agnostic to the actual endpoint - it just calls MCP tools by name.

**Why MCP for everything:**

1. **Unified protocol** - One interface for all external services
2. **Discoverability** - `tools/list` reveals available services
3. **Decoupling** - Environment code doesn't hardcode endpoints
4. **Flexibility** - Training infra maps service names to actual endpoints

## SDK (Optional Helpers)

The SDK provides well-designed helpers that implement best practices. These are optional but strongly recommended.

### RubricComposer

Handles within-environment rubric composition:

```python
from openenv.rewards import RubricComposer

class MyCodingEnvironment(Environment):
    def __init__(self, config: CodingEnvConfig):
        self.composer = RubricComposer(
            rubrics={
                "accuracy": AccuracyRubric(config.accuracy),
                "style": StyleRubric(config.style, self.mcp),
                "safety": SafetyRubric(config.safety, self.mcp),
            },
            weights={
                "accuracy": config.weights.get("accuracy", 1.0),
                "style": config.weights.get("style", 0.5),
                "safety": config.weights.get("safety", 2.0),
            },
            normalize=True,  # Normalize each rubric to [0, 1]
        )

    def step(self, action: Action) -> Observation:
        # ... execute action, get raw observation ...

        # Compute composed reward
        reward, components = self.composer.compute(action, raw_obs)

        return Observation(
            done=done,
            reward=reward,
            metadata={
                "reward_components": components,
                **raw_obs.metadata
            }
        )
```

**RubricComposer Features:**

- Normalizes rubric outputs to consistent scale
- Handles per-turn vs episode-end separation
- Tracks component scores for logging
- Supports weight updates via `update_weights()`

### RewardNormalizer

Handles cross-environment normalization (used by training code):

```python
from openenv.rewards import RewardNormalizer

# Initialize normalizer
normalizer = RewardNormalizer(
    method="running_mean_std",  # or "min_max", "percentile"
    per_environment=True,       # Track stats per env
    clip_range=(-10.0, 10.0),   # Clip extreme values
)

# In training loop
for batch in dataloader:
    obs = env.step(action)

    # Normalize reward
    normalized_reward = normalizer(
        reward=obs.reward,
        env_id=env.environment_id
    )

    # Use normalized reward for learning
    loss = compute_loss(normalized_reward, ...)
```

**Normalization Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| `running_mean_std` | Subtract mean, divide by std | General purpose, PPO-style |
| `min_max` | Scale to [0, 1] based on observed range | When bounds are unknown |
| `percentile` | Robust to outliers | Sparse/noisy rewards |
| `population` | Normalize across all envs in batch | Multi-env training |

### Logging Helpers

Integration with experiment tracking:

```python
from openenv.rewards import RewardLogger

logger = RewardLogger(backend="wandb")  # or "tensorboard", "mlflow"

# In training loop
for step, batch in enumerate(dataloader):
    obs = env.step(action)

    # Log per-rubric scores
    logger.log_step(
        step=step,
        episode=episode_num,
        reward=obs.reward,
        components=obs.metadata.get("reward_components", {}),
        env_id=env.environment_id
    )
```

**Logged Metrics:**

- `reward/total` - Final scalar reward
- `reward/{rubric_name}` - Per-rubric scores
- `reward/weights/{rubric_name}` - Current weights (if dynamic)
- `config/updates` - Config change events

## Multi-Turn Scenarios

For agentic evaluations like Tau-bench with multiple LLM roles:

**Roles:**
- **MUT (Model Under Test)** - The agent being trained/evaluated (orchestrated by training infra)
- **SM (Service Model)** - Simulated user/counterpart (orchestrated by environment)
- **Judge** - Scores the interaction (orchestrated by environment)

**Architecture:**

```
Training Infra                    Environment
      │                                │
      │  action (MUT response)         │
      ├───────────────────────────────►│
      │                                │──► SM (via MCP)
      │                                │◄── SM response
      │                                │
      │                                │──► Judge (via MCP, if episode end)
      │                                │◄── Score
      │  observation (SM response,     │
      │              reward)           │
      │◄───────────────────────────────┤
```

**SM is Stateless:**

The environment manages conversation history. SM receives full history each call:

```python
class TauEnvironment(Environment):
    def __init__(self, config):
        self.mcp = MCPClient(config.mcp_endpoint)
        self.history = []
        self.scenario = None

    def reset(self) -> Observation:
        # Load scenario
        self.scenario = self._sample_scenario()
        self.history = []

        # Get initial SM message
        sm_message = self.mcp.call_tool(
            "service_model",
            persona=self.scenario.user_persona,
            conversation=[],
            instruction="Start the conversation."
        )

        self.history.append({"role": "user", "content": sm_message})
        return Observation(message=sm_message, ...)

    def step(self, action: Action) -> Observation:
        # Add MUT response to history
        self.history.append({"role": "assistant", "content": action.message})

        # Check if conversation should end
        done = self._check_done()

        if not done:
            # Get SM response (full history sent)
            sm_message = self.mcp.call_tool(
                "service_model",
                persona=self.scenario.user_persona,
                conversation=self.history,
                instruction="Continue the conversation."
            )
            self.history.append({"role": "user", "content": sm_message})

        # Compute reward
        reward, components = self._compute_reward(done)

        return Observation(
            message=sm_message if not done else "",
            reward=reward,
            done=done,
            metadata={
                "reward_components": components,
                "conversation_history": self.history,
            }
        )

    def _compute_reward(self, done: bool) -> Tuple[float, dict]:
        components = {}

        # Per-turn rubrics
        for rubric in self.per_turn_rubrics:
            components[rubric.name] = rubric(self.last_action, self.last_obs)

        # Episode-end rubrics (only when done)
        if done:
            for rubric in self.episode_end_rubrics:
                components[rubric.name] = rubric(self.last_action, self.last_obs)

        # Compose
        reward = self.composer.combine(components)
        return reward, components
```

## Reward Shaping and Schedules

Reward shaping helps agents learn in sparse-reward environments but should be reduced over training.

**Example: Chess with Shaping**

Early training:
```yaml
per_turn:
  - name: piece_capture
    rubric: chess.PieceCaptureRubric
    weight: 1.0  # High weight initially

episode_end:
  - name: game_outcome
    rubric: chess.WinLossRubric
    weight: 1.0
```

Later training (via `POST /config`):
```yaml
per_turn:
  - name: piece_capture
    weight: 0.0  # Disabled

episode_end:
  - name: game_outcome
    weight: 1.0  # Only outcome matters
```

**Training Script Pattern:**

```python
# Define schedule
def shaping_weight(episode: int) -> float:
    if episode < 1000:
        return 1.0
    elif episode < 5000:
        return 1.0 - (episode - 1000) / 4000
    else:
        return 0.0

# Training loop
for episode in range(10000):
    # Update weights based on schedule
    if episode % 100 == 0:
        new_weight = shaping_weight(episode)
        env.post("/config", json={
            "per_turn": [{"name": "piece_capture", "weight": new_weight}]
        })
        wandb.log({"reward_shaping_weight": new_weight})

    # Train episode
    ...
```

## Implementation Notes

### For Environment Authors

1. **Use RubricComposer** - Don't hand-roll composition logic
2. **Include reward_components** - Always expose per-rubric scores in metadata
3. **Normalize rubrics** - Each rubric should output [0, 1] or declare its scale
4. **Accept weight updates** - Implement `POST /config` for at least weight changes
5. **Document your rubrics** - What they measure, expected ranges, dependencies

### For Training Framework Authors

1. **Use RewardNormalizer** - Cross-environment normalization is critical
2. **Log per-rubric scores** - Use `metadata["reward_components"]` for debugging
3. **Support schedules** - Allow weight updates via `POST /config`
4. **Track config changes** - Log all config updates for reproducibility

### For Researchers

1. **Version your configs** - Use `schema_version` for reproducibility
2. **Log everything** - Per-rubric scores reveal what's driving learning
3. **Start with shaping** - Then reduce it to test true capability
4. **Compare rubrics** - A/B test different rubric implementations

## Future Work

- **Rubric Registry** - Discover and share rubrics via Hugging Face Hub
- **Automatic Normalization** - SDK infers scales from rubric metadata
- **Curriculum Learning** - SDK support for multi-stage reward schedules
- **Rubric Ensembles** - Combine multiple judges for robust scoring

## References

- RFC 000: Design Principles and Broad Roadmap
- RFC 001: OpenEnv Basic Abstractions
- RFC 002: Framework Spec for Agent Execution Environments
- RFC 003: MCP Support
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Training Language Models with Language Feedback](https://arxiv.org/abs/2204.14146)

## Appendix A: Complete Example

A full example environment with reward pipeline:

```python
# coding_env/server/environment.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openenv.core import Environment, Action, Observation
from openenv.rewards import RubricComposer
from openenv.mcp import MCPClient

@dataclass
class CodingEnvConfig:
    schema_version: str = "1.0"

    # MCP endpoint for external services
    mcp_endpoint: str = "http://localhost:8080"

    # Rubric weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "test_pass": 1.0,
        "code_style": 0.3,
        "efficiency": 0.2,
    })

    # Rubric-specific configs
    test_pass: Dict = field(default_factory=lambda: {
        "timeout": 5.0,
        "partial_credit": True,
    })

    code_style: Dict = field(default_factory=lambda: {
        "llm_service": "style_judge",
    })


class CodingEnvironment(Environment):
    def __init__(self, config: CodingEnvConfig):
        self.config = config
        self.mcp = MCPClient(config.mcp_endpoint)

        # Initialize rubrics
        self.composer = RubricComposer(
            rubrics={
                "test_pass": TestPassRubric(config.test_pass),
                "code_style": StyleRubric(config.code_style, self.mcp),
                "efficiency": EfficiencyRubric(),
            },
            weights=config.weights,
            normalize=True,
        )

        self.current_task = None
        self.submission_history = []

    def reset(self) -> Observation:
        self.current_task = self._sample_task()
        self.submission_history = []

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task": self.current_task.description,
                "test_cases": self.current_task.public_tests,
            }
        )

    def step(self, action: Action) -> Observation:
        # Execute submitted code
        execution_result = self._execute_code(action.code)
        self.submission_history.append(action.code)

        # Build observation
        obs = Observation(
            done=execution_result.all_tests_pass or len(self.submission_history) >= 5,
            reward=0.0,  # Will be set by composer
            metadata={
                "execution_output": execution_result.output,
                "test_results": execution_result.test_results,
                "submission_history": self.submission_history,
            }
        )

        # Compute reward
        reward, components = self.composer.compute(action, obs)
        obs.reward = reward
        obs.metadata["reward_components"] = components

        return obs

    def update_config(self, updates: dict) -> dict:
        """Handle POST /config requests."""
        if "weights" in updates:
            self.config.weights.update(updates["weights"])
            self.composer.update_weights(self.config.weights)

        return {"status": "ok", "config": self.config}

    def get_config(self) -> dict:
        """Handle GET /config requests."""
        return asdict(self.config)
```

## Appendix B: MCP Service Wrapper

Example MCP server wrapping an OpenAI-compatible LLM for judging:

```python
# mcp_services/style_judge.py

from mcp.server import Server
from openai import OpenAI

server = Server("style-judge")
llm = OpenAI(base_url="http://vllm:8000/v1")

JUDGE_PROMPT = """
You are a code style judge. Evaluate the following code for:
- Readability
- Proper naming conventions
- Appropriate comments
- Clean structure

Code:
{code}

Respond with a JSON object: {{"score": 0.0-1.0, "reasoning": "..."}}
"""

@server.tool()
def judge_style(code: str) -> dict:
    """Judge code style quality."""
    response = llm.chat.completions.create(
        model="llama-3.1-70b-instruct",
        messages=[
            {"role": "user", "content": JUDGE_PROMPT.format(code=code)}
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return {
        "score": float(result["score"]),
        "reasoning": result["reasoning"]
    }

if __name__ == "__main__":
    server.run()
```
