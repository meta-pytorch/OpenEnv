# Runtime Providers

A runtime provider starts an environment server and returns a `base_url` that an
`EnvClient` connects to. They all implement the same `ContainerProvider` contract,
so switching from local Docker to a cloud sandbox is a one-line change.

## Available providers

| Provider | Backend | Install |
|----------|---------|---------|
| `LocalDockerProvider` | Local Docker daemon | core |
| `DockerSwarmProvider` | Docker Swarm cluster | core |
| `KubernetesProvider` | Kubernetes cluster | core |
| `UVProvider` | Local process via `uv` (no container) | core |
| `DaytonaProvider` | Daytona cloud sandboxes | `pip install openenv[daytona]` |
| `ACASandboxProvider` | Azure Container Apps Sandboxes | `pip install openenv[aca]` |

Cloud-provider SDKs are optional extras, imported lazily, so installing core
OpenEnv pulls in no cloud SDK. Import a cloud provider from its module:

```python
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider
```

See the [Core API reference](../reference/core.md#container-providers) for each
provider's full API.

## Selecting a provider

`from_docker_image` uses `LocalDockerProvider` by default. Pass `provider=` to
run the server somewhere else:

```python
from openenv.core.containers.runtime.daytona_provider import DaytonaProvider
from tbench2_env import Tbench2Action, Tbench2Env

image = DaytonaProvider.image_from_dockerfile("envs/tbench2_env/server/Dockerfile")
provider = DaytonaProvider()
base_url = provider.start_container(image=image)
provider.wait_for_ready(base_url, timeout_s=180)

try:
    async with Tbench2Env(base_url=base_url, provider=provider) as env:
        result = await env.reset()
        result = await env.step(Tbench2Action(action_type="exec", command="ls -la"))
        print(result.observation.output)
finally:
    provider.stop_container()
```

Full examples: [`examples/daytona_tbench2_simple.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_simple.py)
and [`examples/daytona_tbench2_concurrent.py`](https://github.com/huggingface/OpenEnv/blob/main/examples/daytona_tbench2_concurrent.py).
