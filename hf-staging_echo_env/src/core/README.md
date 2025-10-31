# <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style simple APIs. OpenEnv provides a standard for interacting with agentic execution environments via simple Gymnasium style APIs - step(), reset(), state(). Users of agentic execution environments can interact with the environment during RL training loops using these simple APIs.

In addition to making it easier for researchers and RL framework writers, we also provide tools for environment creators making it easier for them to create richer environments and make them available over familar protocols like HTTP and packaged using canonical technologies like docker. Environment creators can use the OpenEnv framework to create environments that are isolated, secure, and easy to deploy and use.


## Overview
`openenv-core` provides the foundational building blocks for creating and interacting with containerized environments over HTTP. It enables you to build agent environments that can be deployed as Docker containers and accessed via a simple HTTP API.

> ⚠️ **Early Development Warning** OpenEnv is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.


# OpenEnv Core

Core components for OpenEnv - a framework for building HTTP-based agentic environments.

## Features

- **HTTPEnvClient**: Generic HTTP client for interacting with remote environments
- **HTTPEnvServer**: FastAPI-based server wrapper for exposing environments over HTTP
- **Container Providers**: Pluggable architecture for running containers (Docker, Kubernetes, etc.)
- **Type System**: Strongly-typed Action/Observation/State interfaces
- **Web Interface**: Optional web UI for interacting with environments

## Installation

```bash
pip install openenv-core
```

For development:
```bash
pip install openenv-core[dev]
```

## Quick Start

### Creating an Environment Client

```python
from openenv_core import HTTPEnvClient, StepResult
from dataclasses import dataclass

@dataclass
class MyAction:
    text: str

@dataclass
class MyObservation:
    response: str

class MyEnvClient(HTTPEnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> dict:
        return {"text": action.text}

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs_data = payload["observation"]
        return StepResult(
            observation=MyObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> Any:
        return payload

# Use with Docker
env = MyEnvClient.from_docker_image("my-env:latest")
result = env.reset()
step_result = env.step(MyAction(text="hello"))
env.close()
```

### Creating an Environment Server

```python
from openenv_core.env_server import Environment, HTTPEnvServer, create_app
from dataclasses import dataclass

@dataclass
class MyAction:
    text: str

@dataclass
class MyObservation:
    response: str
    reward: float = 0.0
    done: bool = False

class MyEnvironment(Environment):
    def reset(self) -> MyObservation:
        return MyObservation(response="Ready")

    def step(self, action: MyAction) -> MyObservation:
        return MyObservation(
            response=f"Echo: {action.text}",
            reward=1.0,
            done=False
        )

# Create FastAPI app
env = MyEnvironment()
app = create_app(env, MyAction, MyObservation)

# Run with: uvicorn module:app --host 0.0.0.0 --port 8000
```

## Container Providers

OpenEnv Core supports multiple container providers:

### Local Docker Provider

```python
from openenv_core.containers.runtime import LocalDockerProvider

provider = LocalDockerProvider()
base_url = provider.start_container("my-env:latest")
provider.wait_for_ready(base_url)
# Use environment...
provider.stop_container()
```

### Kubernetes Provider (Coming Soon)

```python
from openenv_core.containers.runtime import KubernetesProvider

provider = KubernetesProvider(namespace="envs")
base_url = provider.start_container("my-env:latest")
# Use environment...
provider.stop_container()
```


## API Reference

### HTTPEnvClient

Base class for environment clients with these abstract methods:

- `_step_payload(action)`: Convert action to JSON
- `_parse_result(payload)`: Parse response to StepResult
- `_parse_state(payload)`: Parse state response

### HTTPEnvServer

Server wrapper with these methods:

- `register_routes(app)`: Register endpoints on FastAPI app
- `_deserialize_action(data)`: Convert JSON to Action
- `_serialize_observation(obs)`: Convert Observation to JSON

### Environment Interface

Base interface for environment implementations:

- `reset()`: Reset environment and return initial observation
- `step(action)`: Execute action and return observation
- `state`: Property returning current environment state

## License

This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please see the main OpenEnv repository for contribution guidelines.

## Links

- **Homepage**: https://github.com/facebookresearch/OpenEnv
- **Documentation**: https://github.com/facebookresearch/OpenEnv/blob/main/README.md
- **Bug Tracker**: https://github.com/facebookresearch/OpenEnv/issues
