On this page we will walk you through the process of using an OpenEnv environment. If you want to build your own environment, please see the [Building an Environment](environment-builder.md) page.

## Installation

To install the OpenEnv package, you can use the following command:

```bash
pip install openenv-core
```

!!! note
    This installs both the `openenv` CLI and the `openenv.core` runtime. Environment projects can depend on `openenv-core[core]` if they only need the server/client libraries.

### Using the Echo Environment (Example)

Let's start by using the Echo Environment. This is a simple environment that echoes back messages.

Install the echo environment client package:

```bash
pip install git+https://huggingface.co/spaces/openenv/echo-env 
```

Then you can use the environment via its HTTP interface.

```python
from echo_env import EchoAction, EchoEnv

# Automatically start container and connect
client = EchoEnv(base_url="https://openenv-echo-env.hf.space")

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

### Using environments from Hugging Face

You can also use environments from Hugging Face. To do this, you can use the `from_hub` method of the environment class.

```python
from echo_env import EchoEnv

client = EchoEnv.from_hub("meta-pytorch/echo_env")
```

In the background, the environment will be pulled from Hugging Face and a container will be started on your local machine.

You can also connect to the remote space on Hugging Face by passing the base URL to the environment class.

```python
from echo_env import EchoEnv

client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
```

### Using Docker containers

You can also use environments from Docker containers. To do this, you can use the `from_docker_image` method of the environment class.

```python
from echo_env import EchoEnv

client = EchoEnv.from_docker_image("registry.hf.space/openenv-echo-env:latest")
```

In the background, the environment will be pulled from Docker Hub and a container will be started on your local machine.

As above, you can also connect to the docker container by passing the base URL to the environment class.

```sh
docker run -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

Then you can use the environment via its HTTP interface.

```python
from echo_env import EchoEnv

client = EchoEnv(base_url="http://localhost:8000")
```

### Using AutoEnv and AutoAction (Recommended)

The `AutoEnv` and `AutoAction` classes provide a HuggingFace-style auto-discovery API that automatically selects and instantiates the correct environment client and action classes without manual imports.

```python
from openenv import AutoEnv, AutoAction

# Load environment from installed package
env = AutoEnv.from_env("echo-env")

# Get the action class
EchoAction = AutoAction.from_env("echo-env")

# Use them together
result = env.reset()
result = env.step(EchoAction(message="Hello!"))
print(result.observation.echoed_message)  # "Hello!"

env.close()
```

AutoEnv supports multiple name formats - all of these work:

```python
env = AutoEnv.from_env("echo")       # Short name
env = AutoEnv.from_env("echo-env")   # With suffix
env = AutoEnv.from_env("echo_env")   # Underscore variant
```

You can also load environments directly from HuggingFace Hub:

```python
# From Hub repo ID - auto-downloads and installs if needed
env = AutoEnv.from_env("meta-pytorch/coding-env")
CodeAction = AutoAction.from_env("meta-pytorch/coding-env")

# If the Space is running, connects directly without local Docker
# If not, falls back to local Docker mode
```

To see all available environments:

```python
AutoEnv.list_environments()
AutoAction.list_actions()
```

### Using environments from a local directory

You can also use environments from a local directory. To do this, navigate to the directory of the environment and start the server.

```bash
cd path/to/echo-env

# manage dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# start the server
uv run server --host 0.0.0.0 --port 8000
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then you can use the environment via its HTTP interface.

```python
from echo_env import EchoEnv

client = EchoEnv(base_url="http://localhost:8000")
```

## Nice work! You've now used an OpenEnv environment.

Your next steps are to:

- [Check out the environments](environments.md)
- [Try out the end-to-end tutorial](https://colab.research.google.com/github/meta-pytorch/OpenEnv/blob/main/examples/OpenEnv_Tutorial.ipynb)
- [Build your own environment](environment-builder.md)
