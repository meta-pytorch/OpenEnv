---
title: Coding Environment Server
emoji: 💻
colorFrom: '#007ACC'
colorTo: '#1E1E1E'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
---

# Coding Environment

A Python code execution environment that runs arbitrary Python code and returns results. Perfect for testing code execution infrastructure and demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Coding environment is through the `CodingEnv` class:

```python
from envs.coding_env import CodeAction, CodingEnv

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

That's it! The `CodingEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .
```

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
from envs.coding_env import CodingEnv

# Connect to existing server
coding_env = CodingEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = coding_env.reset()
result = coding_env.step(CodeAction(code="print('Hello!')"))
```

Note: When connecting to an existing server, `coding_env.close()` will NOT stop the server.

## Development & Testing

### Running the Full Example

Run the complete example that demonstrates the full workflow:

```bash
python3 src/envs/coding_env/client/example_usage.py
```

This example shows:
- Creating an environment from a Docker image
- Resetting and executing code through the environment
- Automatic cleanup with `close()`

## Project Structure

```
coding_env/
├── README.md              # This file
├── models.py              # Action, Observation, and State models
├── client/
│   ├── coding_env_client.py  # CodingEnv client implementation
│   └── example_usage.py      # Usage examples
└── server/
    ├── python_codeact_env.py  # Core environment logic
    ├── app.py                 # FastAPI application
    ├── transforms.py          # Observation transforms
    ├── Dockerfile             # Container image definition
    └── README.md              # Server-specific documentation
```
