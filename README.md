# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs.

## Overview

OpenEnv provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - step(), reset(), state(). Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs.

In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the OpenEnv framework to create environments that are isolated, secure, and easy to deploy and use.

> ⚠️ **Early Development Warning** OpenEnv is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

### RFCs

Below is a list of active and historical RFCs for OpenEnv. RFCs are proposals for major changes or features. Please review and contribute!

- [RFC 001: Baseline API and Interface Specifications](https://github.com/meta-pytorch/OpenEnv/pull/26)

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Client Application                   │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  EchoEnv       │              │  CodingEnv       │   │
│  │ (HTTPEnvClient)│              │  (HTTPEnvClient) │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ HTTP                          │ HTTP
            │ (reset, step, state)          │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)               │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   EchoEnvironment    │    │ PythonCodeActEnv     │   │
│  │ (Environment base)   │    │ (Environment base)   │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Web Interface

OpenEnv includes a built-in web interface for interactive environment exploration and debugging. The web interface provides:

- **Two-Pane Layout**: HumanAgent interaction on the left, state observation on the right
- **Real-time Updates**: WebSocket-based live updates without page refresh
- **Dynamic Forms**: Automatically generated action forms based on environment Action types
- **Action History**: Complete log of all actions taken and their results

The web interface is **conditionally enabled** based on environment variables:

- **Local Development**: Disabled by default for lightweight development
- **Manual Override**: Enable with `ENABLE_WEB_INTERFACE=true`

To use the web interface:

```python
from open_env.core.env_server import create_web_interface_app
from open_env.envs.your_env.models import YourAction, YourObservation
from open_env.envs.your_env.server.your_environment import YourEnvironment

env = YourEnvironment()
app = create_web_interface_app(env, YourAction, YourObservation)
```

When enabled, open `http://localhost:8000/web` in your browser to interact with the environment.

#### 2. Environment (Server-Side)
Base class for implementing environment logic:
- **`reset()`**: Initialize a new episode, returns initial `Observation`
- **`step(action)`**: Execute an `Action`, returns resulting `Observation`
- **`state()`**: Access episode metadata (`State` with episode_id, step_count, etc.)

#### 2. HTTPEnvClient (Client-Side)
Base class for HTTP communication:
- Handles HTTP requests to environment server
- Contains a utility to spin up a docker container locally for the corresponding environment
- Type-safe action/observation parsing

#### 3. Container Providers
Manage container deployment:
- `LocalDockerProvider`: Run containers on local Docker daemon
- `KubernetesProvider`: Deploy to K8s clusters (future)

#### 4. Models
Type-safe data structures:
- `Action`: Base class for environment actions
- `Observation`: Base class for environment observations
- `State`: Episode state tracking
- `StepResult`: Combines observation, reward, done flag

## Project Structure

### For Environment Creators

When building a new environment, create the following structure:

```
src/envs/your_env/
├── __init__.py           # Export YourAction, YourObservation, YourEnv
├── models.py             # Define Action, Observation, State dataclasses
├── client.py             # Implement YourEnv(HTTPEnvClient)
├── README.md             # Document your environment
└── server/
    ├── your_environment.py  # Implement YourEnvironment(Environment)
    ├── app.py               # Create FastAPI app
    └── Dockerfile           # Define container image
```

See [`src/envs/README.md`](src/envs/README.md) for a complete guide on building environments.

### For Environment Users

To use an environment:
1. Import from `envs.your_env`: `from envs.echo_env import EchoAction, EchoEnv`
2. Create client: `client = EchoEnv.from_docker_image("echo-env:latest")`
3. Interact: `client.reset()`, `client.step(action)`, `client.state()`
4. Cleanup: `client.close()`

See example scripts in `examples/` directory.

## Design Principles

1. **Separation of Concerns**: Clear client-server boundaries
2. **Type Safety**: Strongly-typed actions, observations, and state
3. **Container Isolation**: Each environment runs in its own container
4. **Simple APIs**: Minimal, intuitive interfaces

## Quick Start

### Using the Echo Environment(Example)

```python
from envs.echo_env import EchoAction, EchoEnv

# Automatically start container and connect
client = EchoEnv.from_docker_image("echo-env:latest")

# Reset the environment
result = client.reset()
print(result.observation.echoed_message)  # "Echo environment ready!"

# Send messages
result = client.step(EchoAction(message="Hello, World!"))
print(result.observation.echoed_message)  # "Hello, World!"
print(result.reward)  # 1.3 (based on message length)

# Cleanup
client.close()  # Stops and removes container
```

## Requirements

- Python 3.11+
- Docker Desktop or Docker Engine
- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- Requests >= 2.25.0
- smolagents (for coding environment)

## Supported RL Tools
The goal of this project is to support a broad set of open and closed tools to help standardize the agentic RL community. If you have a project that supports OpenEnv environments, please put up a PR to add your tool name along with a link to your documentation.

### torchforge
(coming soon)

### TRL
(coming soon}

### Unsloth
(coming soon)

### SkyRL
(coming soon)

## Example Environments

### Echo Environment
A simple environment that echoes back messages with metadata. Perfect for:
- Testing the HTTP server infrastructure
- Learning the framework basics
- Verifying container deployment

See: [`src/envs/echo_env/README.md`](src/envs/echo_env/README.md)

### Coding Environment
Executes arbitrary Python code in a sandboxed environment. Features:
- Safe code execution using smolagents
- Capture stdout, stderr, and exit codes
- Persistent execution context within episodes
- Error handling with detailed messages

See: [`src/envs/coding_env/README.md`](src/envs/coding_env/README.md)

## Community support & Acknowledgments
This is an open and community centric project. If you would like to add your name here, please put up a pull request and tag @jspisak for review. Ty!!

Supporters include: Meta-PyTorch, Hugging Face, [Surge AI](https://surgehq.ai), Unsloth AI, Reflection AI, vLLM, SkyRL (UC-Berkeley), LightningAI, ..

And we'd also like to acknowledge the team at Farama Foundation as the OpenEnv API was heavily inspired by the work you all have done on Gymnasium. Cheers!

## License

BSD 3-Clause License (see LICENSE file)
