# AutoEnv & AutoAction: Auto-Discovery API

OpenEnv provides a HuggingFace-style auto-discovery API that makes it easy to work with environments without manual imports.

## Overview

The auto-discovery system provides two main classes:

- **`AutoEnv`**: Automatically loads and instantiates environment clients
- **`AutoAction`**: Automatically loads action classes for environments

Both classes work with:
- **Local packages**: Installed via `pip install openenv-<env-name>`
- **HuggingFace Hub**: Environments hosted on HuggingFace Spaces

## Quick Start

### Basic Usage

Instead of manually importing specific environment classes:

```python
# Old way
from envs.coding_env import CodingEnv, CodeAction
env = CodingEnv.from_docker_image("coding-env:latest")
```

You can now use the auto-discovery API:

```python
# New way
from openenv import AutoEnv, AutoAction

# Create environment
env = AutoEnv.from_hub("coding-env")

# Get action class
CodeAction = AutoAction.from_hub("coding-env")

# Use them together
result = env.reset()
action = CodeAction(code="print('Hello, OpenEnv!')")
step_result = env.step(action)
env.close()
```

## AutoEnv API

### `AutoEnv.from_hub(name, **kwargs)`

Create an environment client from a name or HuggingFace Hub repository.

**Parameters:**
- `name`: Environment name or Hub repo ID
  - Local: `"coding"`, `"coding-env"`, `"coding_env"`
  - Hub: `"meta-pytorch/coding-env"`, `"username/env-name"`
- `base_url`: Optional base URL for HTTP connection
- `docker_image`: Optional Docker image name (overrides default)
- `container_provider`: Optional container provider
- `wait_timeout`: Timeout for container startup (default: 30s)
- `env_vars`: Optional environment variables for the container
- `**kwargs`: Additional arguments passed to the client class

**Returns:** Instance of the environment client class

**Examples:**

```python
from openenv import AutoEnv

# From installed package
env = AutoEnv.from_hub("coding-env")

# From HuggingFace Hub
env = AutoEnv.from_hub("meta-pytorch/coding-env")

# With custom configuration
env = AutoEnv.from_hub(
    "coding",
    docker_image="my-coding-env:v2",
    wait_timeout=60.0,
    env_vars={"DEBUG": "1"}
)
```

### `AutoEnv.list_environments()`

List all available environments.

```python
from openenv import AutoEnv

AutoEnv.list_environments()
# Output:
# Available Environments:
# ----------------------------------------------------------------------
# coding         : Coding environment for OpenEnv (v0.1.0)
# echo           : echo_env environment (v0.1.0)
# browsergym     : BrowserGym environment (v0.1.0)
# ...
```

### `AutoEnv.get_env_info(name)`

Get detailed information about an environment.

```python
from openenv import AutoEnv

info = AutoEnv.get_env_info("coding")
print(f"Description: {info['description']}")
print(f"Version: {info['version']}")
print(f"Docker Image: {info['default_image']}")
print(f"Client Class: {info['env_class']}")
print(f"Action Class: {info['action_class']}")
```

### `AutoEnv.get_env_class(name)`

Get the environment class (not an instance).

```python
from openenv import AutoEnv

CodingEnv = AutoEnv.get_env_class("coding")
# Now you can instantiate it yourself with custom parameters
env = CodingEnv.from_docker_image("coding-env:latest", wait_timeout=60.0)
```

## AutoAction API

### `AutoAction.from_hub(name)`

Get the Action class from an environment name or HuggingFace Hub repository.

**Parameters:**
- `name`: Environment name or Hub repo ID

**Returns:** Action class (not an instance!)

**Examples:**

```python
from openenv import AutoAction

# From installed package
CodeAction = AutoAction.from_hub("coding-env")
action = CodeAction(code="print('Hello!')")

# From HuggingFace Hub
CodeAction = AutoAction.from_hub("meta-pytorch/coding-env")

# Different name formats work
EchoAction = AutoAction.from_hub("echo")
EchoAction = AutoAction.from_hub("echo-env")
EchoAction = AutoAction.from_hub("echo_env")
```

### `AutoAction.from_env(env_name)`

Alias for `from_hub()` for clarity when working with environment names.

```python
from openenv import AutoAction

CodeAction = AutoAction.from_env("coding")
action = CodeAction(code="x = 5 + 3")
```

### `AutoAction.list_actions()`

List all available action classes.

```python
from openenv import AutoAction

AutoAction.list_actions()
# Output:
# Available Action Classes:
# ----------------------------------------------------------------------
# coding         : CodeAction
# echo           : EchoAction
# browsergym     : BrowsergymAction
# ...
```

### `AutoAction.get_action_info(name)`

Get detailed information about an action class.

```python
from openenv import AutoAction

info = AutoAction.get_action_info("coding")
print(f"Action Class: {info['action_class']}")
print(f"Module: {info['module']}")
```

## HuggingFace Hub Integration

### Loading from HuggingFace Spaces

AutoEnv can automatically connect to environments running on HuggingFace Spaces:

```python
from openenv import AutoEnv, AutoAction

# Load from HuggingFace Space
env = AutoEnv.from_hub("username/coding-env-test")

# Get action class
CodeAction = AutoAction.from_hub("username/coding-env-test")

# Use normally
result = env.reset()
action = CodeAction(code="print('Hello from HF Space!')")
step_result = env.step(action)

print(f"Output: {step_result.observation.stdout}")
env.close()
```

The system automatically:
1. Detects HuggingFace repo IDs (format: `username/repo-name`)
2. Resolves the Space URL (e.g., `https://username-repo-name.hf.space`)
3. Checks if the Space is running and accessible
4. Downloads the environment package if needed
5. Connects to the running Space

## Complete Workflow Example

Here's a complete example showing the auto-discovery workflow:

```python
from openenv import AutoEnv, AutoAction

# 1. List available environments
print("Available environments:")
AutoEnv.list_environments()

# 2. Create environment
env = AutoEnv.from_hub("coding-env")

# 3. Get action class
CodeAction = AutoAction.from_hub("coding-env")

# 4. Run environment
result = env.reset()
print(f"Environment ready: {result.observation}")

# 5. Execute actions
action = CodeAction(code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci(10) = {fibonacci(10)}")
""")

step_result = env.step(action)
print(f"Output:\n{step_result.observation.stdout}")

# 6. Clean up
env.close()
```

## Error Handling

The auto-discovery API provides helpful error messages:

```python
from openenv import AutoEnv

try:
    env = AutoEnv.from_hub("nonexistent-env")
except ValueError as e:
    print(e)
    # Output:
    # Unknown environment 'nonexistent'.
    # Did you mean: coding?
    # Available environments: atari, browsergym, chat, coding, ...
```

For typos, it suggests similar environment names:

```python
try:
    env = AutoEnv.from_hub("cooding-env")  # Typo
except ValueError as e:
    print(e)
    # Output:
    # Unknown environment 'cooding'.
    # Did you mean: coding?
    # Available environments: ...
```

## Flexible Name Formats

AutoEnv accepts multiple name formats:

```python
from openenv import AutoEnv

# All of these work and refer to the same environment:
env = AutoEnv.from_hub("coding")           # Simple name
env = AutoEnv.from_hub("coding-env")       # With suffix
env = AutoEnv.from_hub("coding_env")       # With underscore
env = AutoEnv.from_hub("coding-env:latest") # With tag (ignored)
```

## How It Works

The auto-discovery system works by:

1. **Package Discovery**: Uses `importlib.metadata` to find installed `openenv-*` packages
2. **Manifest Loading**: Reads `openenv.yaml` files from package resources
3. **Caching**: Caches discovery results for performance
4. **Lazy Loading**: Only imports classes when actually needed
5. **Hub Support**: Downloads and installs packages from HuggingFace Hub on-demand

### Environment Packages

Environments are distributed as installable Python packages:

```bash
# Install an environment
pip install openenv-coding-env

# Now it's automatically discoverable
python -c "from openenv import AutoEnv; AutoEnv.list_environments()"
```

Each environment package includes:
- Client classes (e.g., `CodingEnv`)
- Action/Observation models (e.g., `CodeAction`, `CodeObservation`)
- Server Docker image
- `openenv.yaml` manifest describing the environment

### Manifest Format

Each environment includes an `openenv.yaml` file:

```yaml
name: coding_env
version: 0.1.0
description: Coding environment for OpenEnv

client:
  class_name: CodingEnv
  module: coding_env.client

action:
  class_name: CodeAction
  module: coding_env.client

observation:
  class_name: CodeObservation
  module: coding_env.client

default_image: coding-env:latest
spec_version: 1
```

## Benefits

✅ **Simple**: No need to know which module to import from
✅ **Flexible**: Works with local packages and HuggingFace Hub
✅ **Discoverable**: List and explore available environments
✅ **Type-Safe**: Returns properly typed environment classes
✅ **HuggingFace-style**: Familiar API for ML practitioners
✅ **Performant**: Caching and lazy loading for efficiency

## See Also

- [Environment Builder Guide](environment-builder.md) - How to create your own environments
- [Core API Documentation](core.md) - Low-level API details
- [HuggingFace Hub](https://huggingface.co/meta-pytorch) - Pre-built environments
