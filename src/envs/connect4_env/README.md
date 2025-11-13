---
title: Connect4 Environment Server
emoji: 🔴
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - games
---

# Connect4 Environment

This repository packages the classic Connect4 board game as a standalone OpenEnv
environment. It exposes a FastAPI server compatible with the OpenEnv CLI and
provides a Python client for interacting with the environment programmatically.

## Quick Start

```bash
# Install dependencies (editable mode for local development)
uv pip install -e .

# Launch the server locally
uv run server
```

Once running, visit `http://localhost:8000/docs` to explore the OpenAPI schema.

## Python Usage

```python
from connect4_env import Connect4Env, Connect4Action

env = Connect4Env.from_docker_image("connect4-env:latest")
result = env.reset()
print(result.observation.board)

result = env.step(Connect4Action(column=3))
print(result.reward, result.done)

env.close()
```

## Deploy

- **Validate:** `openenv validate`
- **Build Docker image:** `openenv build`
- **Push to Hugging Face / Docker Hub:** `openenv push`

Customize the Docker build or deployment metadata through environment variables
as needed. The default server listens on port `8000`.