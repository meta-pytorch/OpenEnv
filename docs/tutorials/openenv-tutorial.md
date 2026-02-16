# OpenEnv Tutorial

## Introduction

OpenEnv is a framework for building isolated, type-safe execution environments for agentic RL training. It provides a standard Gymnasium-style API (`step()`, `reset()`, `state()`) for interacting with environments that run in Docker containers, accessible via WebSocket/HTTP.

Key features: parallel execution of multiple environments, one-click deployment to Hugging Face Spaces, container isolation for security, and language-agnostic access via HTTP/WebSocket.

## Quick Start

Install the core package and an environment client:

```bash
# Note: The environment package includes openenv-core as a dependency,
# but we show both commands here for clarity
pip install openenv-core  # Core framework (EnvClient, types)
pip install git+https://huggingface.co/spaces/openenv/echo_env  # Environment client
```

OpenEnv uses a modular architecture: `openenv-core` provides the base framework, while each environment is a separate package with its own client and models. This allows you to install only the environments you need.

**Finding environments:**
- Browse the [openenv organization](https://huggingface.co/openenv) on Hugging Face
- Filter by category: [agent-environment spaces](https://huggingface.co/spaces?category=agent-environment)
- Explore the [Environment Hub collection](https://huggingface.co/collections/openenv/environment-hub)

Use the environment with async API. The `base_url` is the public URL on Hugging Face Spaces:

```python
from echo_env import EchoAction, EchoEnv

async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
    print(result.observation.echoed_message)

    result = await client.step(EchoAction(message="Hello, World!"))
    print(result.observation.echoed_message)
    print(f"Reward: {result.reward}")
```

For more installation options, see the [Quick Start Guide](../quickstart.md).

## Architecture

OpenEnv uses a client-server architecture with isolated containers:

```
Training Code                    Docker Container
┌─────────────────┐             ┌──────────────────┐
│ env = EchoEnv() │ WebSocket   │ FastAPI Server   │
│ env.reset()     │────────────▶│ Environment      │
│ env.step(...)   │◀────────────│ (reset, step)    │
└─────────────────┘             └──────────────────┘
```

**Key benefits:** Type-safe communication, container isolation, language-agnostic API, and parallel execution support.

## Using Environments

### Echo Environment

The simplest environment for testing and learning ([space](https://huggingface.co/spaces/openenv/echo_env)):

```python
from echo_env import EchoAction, EchoEnv

async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()

    for msg in ["Test 1", "Test 2", "Test 3"]:
        result = await client.step(EchoAction(message=msg))
        print(f"Sent: {msg}")
        print(f"  → Echoed: {result.observation.echoed_message}")
        print(f"  → Reward: {result.reward}")
```

**Key Features:**
- Simple message echoing
- Reward based on message length
- Perfect for testing infrastructure

### Wordle (TextArena)

Word guessing game from TextArena ([space](https://huggingface.co/spaces/openenv/wordle)):

```python
from textarena_env import TextArenaAction, TextArenaEnv

async with TextArenaEnv(base_url="https://openenv-wordle.hf.space") as client:
    result = await client.reset()
    print(f"Goal: {result.observation.prompt}")

    guesses = ["crane", "slate", "audio"]
    for guess in guesses:
        result = await client.step(TextArenaAction(message=f"[{guess}]"))

        # Check feedback
        for msg in result.observation.messages:
            print(f"{msg.content}")

        print(f"Reward: {result.reward}")
        if result.done:
            break
```

**Key Features:**
- Classic Wordle gameplay
- Feedback with green/yellow/gray markers
- Reward signals for letter matches
- Part of TextArena's 15+ text games

### BrowserGym (Web Automation)

Web automation with 100+ MiniWoB tasks and realistic WebArena benchmarks ([space](https://huggingface.co/spaces/openenv/browsergym_env)):

```python
from browsergym_env import BrowserGymAction, BrowserGymEnv

async with BrowserGymEnv(base_url="https://openenv-browsergym-env.hf.space") as client:
    result = await client.reset()
    print(f"Goal: {result.observation.goal}")

    # Simple click task
    action = BrowserGymAction(action_str="click('Submit button')")
    result = await client.step(action)

    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")
```

**Key Features:**
- 100+ MiniWoB training tasks (click, form filling, navigation)
- 812 WebArena evaluation tasks (real websites)
- Visual observations with screenshots
- Natural language actions

### More Environments

**Officially supported environments** (maintained in the [OpenEnv repository](https://github.com/meta-pytorch/OpenEnv/tree/main/envs)):

- **Web Automation**: BrowserGym, OpenApp
- **Games**: Chess, Connect4, Snake, Unity, Atari
- **Text/Language**: TextArena, Wordle
- **Code Execution**: Coding, Git, REPL, Julia
- **Research**: T-Bench2, WebSearch, FinRL, SumoRL
- **And more...**

**Community environments**: Any OpenEnv-compatible environment can be used, whether from the [openenv organization](https://huggingface.co/openenv), the [Environment Hub collection](https://huggingface.co/collections/openenv/environment-hub), the [agent-environment category](https://huggingface.co/spaces?category=agent-environment), or third-party repositories

## Creating Your Own Environment

Creating a custom environment involves defining type-safe models, implementing the environment logic, and containerizing it for deployment. The process is straightforward with the OpenEnv CLI.

For a complete step-by-step guide, see the [Environment Builder Guide](../environment-builder.md).

## Deployment

### Hugging Face Spaces (Recommended)

Deploy your environment to Hugging Face Spaces with zero infrastructure management:

```bash
cd my_env
openenv push --repo-id your-username/my-env
```

This automatically:
- Creates a Space with Docker SDK
- Builds and deploys the container
- Provides a public URL with automatic scaling

Connect to deployed environment:

```python
from my_env import MyAction, MyEnv

with MyEnv(base_url="https://your-username-my-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(MyAction(command="test"))
```

### Local Docker (Development)

For local testing, build and run containers manually:

```bash
docker build -t my-env:latest -f my_env/server/Dockerfile .
docker run -p 8000:8000 my-env:latest
```

## Training Integration

OpenEnv environments work with popular RL frameworks:

- **TRL** (Hugging Face): Most mature integration for GRPO training → [Guide](https://huggingface.co/docs/trl/openenv)
- **torchforge** (PyTorch): [GRPO BlackJack example](https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack)
- **SkyRL** (UC Berkeley): [Documentation](https://skyrl.readthedocs.io/en/latest/examples/openenv.html)
- **Unsloth**: [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb)
- **ART**: [Integration guide](https://art.openpipe.ai/integrations/openenv-integration)
- **Oumi**: [Example notebook](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20OpenEnv%20GRPO%20with%20trl.ipynb)

## Best Practices

For best practices on type safety, error handling, state management, and testing, see the [Environment Builder Guide](../environment-builder.md#best-practices).

## Resources

- **GitHub**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Documentation**: [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/)
- **Discord**: [Join the community](https://discord.gg/YsTYBh6PD9)
- **Examples**: [github.com/meta-pytorch/OpenEnv/examples](https://github.com/meta-pytorch/OpenEnv/tree/main/examples)
- **Environments**: [Available environments](../environments.md)

## Next Steps

1. Try the [Quick Start Guide](../quickstart.md)
2. Explore [available environments](../environments.md)
3. Build your own with the [Environment Builder Guide](../environment-builder.md)
4. Check out [training examples](https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack)
5. Join the [Discord community](https://discord.gg/YsTYBh6PD9)
