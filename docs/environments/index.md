# Environments

OpenEnv provides a growing collection of ready-to-use environments for agentic RL training. Each environment follows the standard OpenEnv API (`step()`, `reset()`, `state()`) and can be deployed via Docker or connected to directly.

## Quick Start

```python
from openenv import AutoEnv

# Load any environment by name
env = AutoEnv("echo")  # Simple echo environment for testing
env = AutoEnv("chess")  # Chess game environment
env = AutoEnv("coding")  # Code execution environment
```

## Environments by Category

### :material-gamepad-variant: Games & Puzzles

Classic games and puzzles for training game-playing agents.

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| [Echo](echo.md) | Simple echo environment for testing | Beginner |
| [Snake](snake.md) | Classic snake game | Beginner |
| [Chess](chess.md) | Chess with Stockfish integration | Intermediate |
| [Atari](atari.md) | Atari games via ALE | Intermediate |
| [OpenSpiel](openspiel.md) | DeepMind's game library | Advanced |
| [TextArena](textarena.md) | Text-based competitive games | Intermediate |

### :material-code-braces: Code Execution

Environments for training coding agents.

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| [Coding](coding.md) | Execute code in isolated containers | Intermediate |
| [REPL](repl.md) | Interactive Python REPL | Beginner |
| [Git](git.md) | Git repository operations | Intermediate |
| [Terminal-Bench 2](tbench2.md) | Terminal command benchmarks | Advanced |

### :material-web: Web & Browser

Web interaction and browser automation environments.

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| [BrowserGym](browsergym.md) | Browser automation with Playwright | Advanced |
| [OpenApp](openapp.md) | Mobile app interaction | Advanced |
| [Web Search](websearch.md) | Web search and retrieval | Intermediate |

### :material-robot: Simulation

Physics and world simulation environments.

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| [Unity](unity.md) | Unity ML-Agents integration | Advanced |
| [SUMO](sumo.md) | Traffic simulation | Advanced |
| [FinRL](finrl.md) | Financial trading simulation | Advanced |

### :material-chat: Conversational

Dialogue and conversation environments.

| Environment | Description | Difficulty |
|-------------|-------------|------------|
| [Chat](chat.md) | General chat environment | Beginner |
| [DIPG](dipg.md) | Safety-focused dialogue | Intermediate |

## Contributing an Environment

Want to add your own environment? See the [Building Environments](../guides/first-environment.md) guide to get started.
