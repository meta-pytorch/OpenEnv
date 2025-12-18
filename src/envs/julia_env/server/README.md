# Julia Environment Server

HTTP server for executing Julia code with test result tracking and reward calculation.

## Overview

This server provides a Julia code execution environment through OpenEnv's HTTP interface. It executes Julia code, parses test results from the `Test` module, and calculates rewards based on execution success and test outcomes.

## Features

- ✅ Execute Julia code in isolated subprocess
- ✅ Parse `Test` module output (tests passed/failed)
- ✅ Calculate rewards based on execution results
- ✅ Safety transforms for output truncation
- ✅ Docker support for reproducible execution
- ✅ Compatible with GRPO training

## Docker Setup

### Prerequisites

First, build the OpenEnv base image (one-time setup):

```bash
# From OpenEnv root directory
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
```

### Build Julia Environment Image

```bash
# From OpenEnv root directory
docker build -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .
```

### Run the Server

```bash
# Run in background with default settings (port 8000, 4 workers)
docker run -d -p 8000:8000 --name julia-env-server julia-env:latest

# OR run in foreground (to see logs)
docker run -p 8000:8000 --name julia-env-server julia-env:latest

# Run with custom port
docker run -d -p 9000:9000 -e PORT=9000 --name julia-env-server julia-env:latest

# Run with custom number of workers (uvicorn workers)
docker run -d -p 8000:8000 -e NUM_WORKER=8 --name julia-env-server julia-env:latest

# Run with custom Julia max workers (for process pool)
docker run -d -p 8000:8000 -e JULIA_MAX_WORKERS=32 --name julia-env-server julia-env:latest

# Run with all custom configurations
docker run -d -p 9000:9000 \
  -e PORT=9000 \
  -e NUM_WORKER=8 \
  -e JULIA_MAX_WORKERS=32 \
  --name julia-env-server julia-env:latest
```

### Test the Server

```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"healthy"}

# Check Julia version inside container
docker exec julia-env-server julia --version
# Expected: julia version 1.10.0
```

### Docker Management Commands

```bash
# View logs
docker logs julia-env-server
docker logs -f julia-env-server  # Follow logs

# Stop/start container
docker stop julia-env-server
docker start julia-env-server

# Remove container
docker rm -f julia-env-server

# Rebuild after code changes
docker build -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .
docker rm -f julia-env-server
docker run -d -p 8000:8000 --name julia-env-server julia-env:latest

# Interactive debugging
docker exec -it julia-env-server /bin/bash
```

## Local Development (Without Docker)

### Prerequisites

- Python 3.10+
- Julia 1.10.0+ installed and in PATH
- FastAPI and dependencies

### Install Julia

**Using juliaup (recommended):**
```bash
curl -fsSL https://install.julialang.org | sh
```

**Or download from:** https://julialang.org/downloads/

### Install Python Dependencies

```bash
pip install fastapi uvicorn
```

### Run Server Locally

```bash
# From OpenEnv root directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
python -m envs.julia_env.server.app
```

Server will start at: http://localhost:8000

## API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy"}
```

### Reset Environment
```
POST /reset
Response: {
  "observation": {
    "stdout": "",
    "stderr": "",
    "exit_code": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "reward": 0.0,
    "execution_time": 0.0
  }
}
```

### Execute Code (Step)
```
POST /step
Body: {"code": "function add(a,b)\n  a+b\nend\nusing Test\n@test add(2,3)==5"}
Response: {
  "observation": {
    "stdout": "Test Passed",
    "stderr": "",
    "exit_code": 0,
    "tests_passed": 1,
    "tests_failed": 0,
    "reward": 1.0,
    "execution_time": 0.15
  },
  "reward": 1.0,
  "done": false
}
```

### Get State
```
GET /state
Response: {
  "episode_id": "uuid",
  "step_count": 5,
  "last_exit_code": 0,
  "total_tests_passed": 10,
  "total_tests_failed": 2
}
```

## Reward Structure

The environment calculates rewards based on:

- **Failed execution** (exit_code != 0): `-0.5`
- **Clean execution** (exit_code == 0): `+0.2`
- **Tests passed**: `+0.3 × (passed/total)`
- **Tests failed**: `-0.2 × (failed/total)`
- **All tests passed bonus**: `+0.5`

Example:
```julia
# 3 tests pass, 1 fails → exit_code 1
reward = -0.5  # Failed execution
# Total: -0.5

# 3 tests pass, 0 fail → exit_code 0
reward = 0.2 + 0.3 × 1.0 + 0.5 = 1.0
# Total: 1.0 (perfect score!)
```

## Test Parsing

The environment parses Julia's `Test` module output:

### Method 1: Error Message Pattern
```
Some tests did not pass: 3 passed, 1 failed, 0 errored, 0 broken.
→ tests_passed=3, tests_failed=1
```

### Method 2: Test Summary Table
```
Test Summary:      | Pass  Fail  Total  Time
Add function Tests |    3     1      4  0.5s
→ tests_passed=3, tests_failed=1
```

## Example Usage

### From Python Client

```python
from envs.julia_env import JuliaEnv, JuliaAction

# Connect to server
env = JuliaEnv(base_url="http://localhost:8000")

# Reset
result = env.reset()

# Execute Julia code with tests
core_code = """
function fibonacci(n)
    if n <= 1
        return n
    end
    return fibonacci(n-1) + fibonacci(n-2)
end
"""

test_code = """
using Test
@test fibonacci(0) == 0
@test fibonacci(1) == 1
@test fibonacci(5) == 5
@test fibonacci(10) == 55
"""

result = env.step(JuliaAction(core_code=core_code, test_code=test_code))

print(f"Exit code: {result.observation.exit_code}")
print(f"Tests passed: {result.observation.tests_passed}")
print(f"Tests failed: {result.observation.tests_failed}")
print(f"Reward: {result.reward}")

# Close connection
env.close()
```

### Example Script

```bash
# From OpenEnv root
python examples/julia_simple.py
```

## GRPO Training Integration

This environment is designed for GRPO (Group Relative Policy Optimization) training:

```python
# In your GRPO training loop
async def play_julia_game(game_idx, game_id, server_url, policy, tokenizer):
    env = JuliaEnv(base_url=server_url)

    # Generate code with LLM
    prompt = format_julia_prompt(task)
    responses = await policy.generate.route(prompt)
    core_code = extract_julia_code(responses[0].text)
    test_code = extract_test_code(responses[0].text)

    # Execute in environment
    result = env.step(JuliaAction(core_code=core_code, test_code=test_code))

    # Get reward
    reward = result.observation.reward

    return {
        "prompt": prompt,
        "response": responses[0],
        "reward": reward,
        "tests_passed": result.observation.tests_passed,
        "tests_failed": result.observation.tests_failed
    }
```

See `examples/grpo_blackjack/` for a complete GRPO training example that can be adapted for Julia.

## Configuration

### Docker Environment Variables

The Docker container accepts the following environment variables:

- **`PORT`**: HTTP server port (default: `8000`)
  - Controls which port the FastAPI server listens on
  - Must match the port mapping in `-p` flag (e.g., `-p 9000:9000 -e PORT=9000`)

- **`NUM_WORKER`**: Number of uvicorn worker processes (default: `4`)
  - Controls parallel request handling capacity
  - More workers = more concurrent requests but higher memory usage
  - Recommended: 2-8 workers for typical workloads

- **`JULIA_MAX_WORKERS`**: Maximum Julia process pool size (default: `16`)
  - Controls maximum concurrent Julia code executions
  - Higher values allow more parallel Julia executions
  - Each worker consumes memory; tune based on available resources
  - Recommended: 8-32 workers depending on your workload

### Runtime Environment Variables

These can be set when running locally (non-Docker):

- `HOST`: Server host (default: 0.0.0.0)
- `JULIA_TIMEOUT`: Julia execution timeout in seconds (default: 60)

### Dockerfile Customization

To use a different Julia version:

```dockerfile
# In Dockerfile, change the version
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11
```

## Troubleshooting

### Julia not found
```bash
# Verify Julia is in PATH
julia --version

# In Docker, check installation
docker exec julia-env-server julia --version
```

### Port already in use
```bash
# Use different port
docker run -p 8001:8000 --name julia-env-server julia-env:latest

# Update client base_url
env = JuliaEnv(base_url="http://localhost:8001")
```

### Container exits immediately
```bash
# Check logs
docker logs julia-env-server

# Run in foreground to see errors
docker run -p 8000:8000 julia-env:latest
```

### Build failures
```bash
# Clean build with no cache
docker build --no-cache -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .

# Verbose output
docker build --progress=plain -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .
```

## Architecture

```
┌─────────────────────────────────────┐
│   Python Client (HTTP)              │
│   JuliaEnv                          │
└────────────┬────────────────────────┘
             │ HTTP POST /step
             │ {"code": "..."}
             ▼
┌─────────────────────────────────────┐
│   FastAPI Server                    │
│   app.py                            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   JuliaCodeActEnv                   │
│   - Execute code via JuliaExecutor  │
│   - Parse test results              │
│   - Calculate rewards               │
│   - Apply transforms                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   JuliaExecutor (subprocess)        │
│   - Write code to temp file         │
│   - Run: julia temp_file.jl         │
│   - Capture stdout/stderr           │
│   - Return results                  │
└─────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/envs/julia_env/

# Integration test
python examples/julia_simple.py
```

### Code Structure

```
server/
├── Dockerfile              # Docker build instructions
├── README.md              # This file
├── __init__.py            # Package initialization
├── app.py                 # FastAPI server entry point
├── julia_codeact_env.py   # Environment implementation
└── julia_transforms.py    # Output transforms
```

## License

BSD-style license. See LICENSE file in repository root.
