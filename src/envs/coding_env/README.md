---
title: Coding Environment Server
emoji: 💻
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Coding Environment

A Python code execution environment that runs arbitrary Python code and returns results. Perfect for testing code execution infrastructure and demonstrating environment usage patterns.

## Installation & Usage

The Coding Environment supports two usage modes:

### Mode 1: In-Repository Development (Recommended for Contributors)

Use this mode when developing or contributing to OpenEnv.

**Setup:**
```bash
# 1. Clone the repository
git clone https://github.com/facebookresearch/OpenEnv.git
cd OpenEnv

# 2. Install in development mode
pip install -e .

# 3. Build the Docker image (from repo root)
docker build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .

# 4. Run the example
python ./examples/local_coding_env.py
```

**Code example:**
```python
# Use in-repo import paths
from envs.coding_env import CodeAction, CodingEnv

try:
    # Create environment from Docker image
    coding_env = CodingEnv.from_docker_image("coding-env:latest")

    # Execute Python code
    result = coding_env.step(CodeAction(code="print('Hello, World!')"))
    print(f"stdout: {result.observation.stdout.strip()}")
    print(f"exit_code: {result.observation.exit_code}")
finally:
    coding_env.close()
```

### Mode 2: Standalone Package (For End Users)

Use this mode when using coding_env as a standalone package.

**Setup:**
```bash
# 1. Install openenv-core (once available on PyPI)
pip install openenv-core

# 2. Install coding_env package
pip install openenv-coding_env

# 3. Use the same Docker image as in-repo mode
# The client-server communicate over HTTP, so the Docker build mode doesn't matter for testing
# You can use the in-repo built image: coding-env:latest
```

**Code example:**
```python
# Use standalone import paths
from coding_env import CodeAction, CodingEnv

try:
    # Connect to the same Docker image built in in-repo mode
    coding_env = CodingEnv.from_docker_image("coding-env:latest")
    result = coding_env.step(CodeAction(code="print('Hello, World!')"))
    print(f"stdout: {result.observation.stdout.strip()}")
finally:
    coding_env.close()
```

## Quick Start Example

**In-repo mode:**
```bash
# From OpenEnv repo root, after pip install -e .
python ./examples/local_coding_env.py
```

**Standalone mode:**

For standalone testing, use a separate test script (the repo example uses in-repo imports only):

```python
# save as test_standalone.py
from coding_env import CodeAction, CodingEnv

try:
    # Uses the same Docker image as in-repo mode
    client = CodingEnv.from_docker_image("coding-env:latest")
    result = client.step(CodeAction(code="print('Hello from standalone!')"))
    print(f"stdout: {result.observation.stdout.strip()}")
finally:
    client.close()
```

**Note:** The client (your Python code) and server (Docker container) are independent. The standalone client can connect to the in-repo Docker image because they communicate over HTTP.

### Manual Usage Example

Once set up (either mode), the usage is identical:

```python
from coding_env import CodeAction, CodingEnv  # or: from envs.coding_env import ...

try:
    # Create environment from Docker image
    coding_env = CodingEnv.from_docker_image("coding-env:latest")

    # Reset
    result = coding_env.reset()
    print(f"Reset complete: exit_code={result.observation.exit_code}")

    # Execute Python code
    code_samples = [
        "print('Hello, World!')",
        "x = 5 + 3\nprint(f'Result: {x}')",
        "import math\nprint(math.pi)"
    ]

    for code in code_samples:
        result = coding_env.step(CodeAction(code=code))
        print(f"Code: {code}")
        print(f"  → stdout: {result.observation.stdout.strip()}")
        print(f"  → exit_code: {result.observation.exit_code}")

finally:
    # Always clean up
    coding_env.close()
```

The `CodingEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Environment Details

### Action
**CodeAction**: Contains a single field
- `code` (str) - The Python code to execute

### Observation
**CodeObservation**: Contains the execution results
- `stdout` (str) - Standard output from code execution
- `stderr` (str) - Standard error from code execution
- `exit_code` (int) - Exit code (0 for success, non-zero for errors)

### State
**CodeState**: Tracks execution state
- `episode_id` (str) - Unique identifier for the episode
- `step_count` (int) - Number of steps taken
- `last_exit_code` (int) - Exit code from the last execution

## Advanced Usage

### Connecting to an Existing Server

If you already have a Coding environment server running, you can connect directly:

```python
# In-repo mode
from envs.coding_env import CodingEnv

# OR standalone mode
from coding_env import CodingEnv

# Connect to existing server
coding_env = CodingEnv(base_url="http://localhost:8000")

# Use as normal
result = coding_env.reset()
result = coding_env.step(CodeAction(code="print('Hello!')"))
```

Note: When connecting to an existing server, `coding_env.close()` will NOT stop the server.

## Docker Build Options

The Dockerfile supports two build modes:

```bash
# In-repo build (default) - from OpenEnv repo root
docker build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .

# Standalone build - from coding_env package directory (for distribution only)
docker build -t coding-env:standalone -f server/Dockerfile --build-arg BUILD_MODE=standalone .
```

**When to use each mode:**

- **In-repo mode (default)**: For development and testing (works with both client modes)
- **Standalone mode**: Only needed when distributing the Docker image without the full OpenEnv repo

**Important:** For local testing, the in-repo Docker image works with both in-repo and standalone clients. The client and server communicate over HTTP, so they're independent. The BUILD_MODE distinction is primarily for distribution/packaging purposes.

## Development & Testing

### Running Tests

```bash
# From repo root
pytest tests/envs/test_python_codeact_reset.py
```

### Building Packages Locally

```bash
# Build openenv-core
cd src
python -m build -w

# Build coding_env
cd envs/coding_env
python -m build -w
```

## Project Structure

```
coding_env/
├── README.md                  # This file
├── pyproject.toml             # Package configuration
├── __init__.py                # Package exports
├── models.py                  # Action, Observation, and State models
├── client.py                  # CodingEnv client implementation
└── server/
    ├── python_codeact_env.py  # Core environment logic
    ├── python_executor.py     # Code execution wrapper
    ├── app.py                 # FastAPI application
    ├── transforms.py          # Observation transforms
    ├── Dockerfile             # Container image (dual-mode)
    └── README.md              # Server-specific documentation
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
- **In-repo mode**: Make sure you ran `pip install -e .` from the repo root
- **Standalone mode**: Check imports use `from coding_env import...` (not `from envs.coding_env import...`)

### Docker Build Failures

- **In-repo**: Build from OpenEnv repo root, not from coding_env directory
- **Standalone**: Only needed for distribution; for testing, use the in-repo build

### Container Won't Start

Check logs without `--rm`:
```bash
docker run -d --name debug-coding-env -p 8765:8000 coding-env:latest
docker logs debug-coding-env
```

### "Do I need to build Docker differently for standalone testing?"

**No!** The in-repo Docker build (`coding-env:latest`) works perfectly with both:
- In-repo client: `from envs.coding_env import CodingEnv`
- Standalone client: `from coding_env import CodingEnv`

The client and server communicate over HTTP, so they're independent. BUILD_MODE is only for distribution.

