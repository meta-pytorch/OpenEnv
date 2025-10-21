# RFC: OpenEnv Framework Spec for agent execution environments

**Status**: In Review
**Created**: 10/14/2025
**Authors**: @Darktex, @pankit-eng, @jspisak, @zkwentz
**RFC ID:** 002

## Summary

An e2e framework for creating, deploying and using isolated execution environments for agentic RL training, built using Gymnasium style APIs. It provides a clean client-server architecture where environments run as FastAPI servers in Docker containers, and clients interact with them via type-safe HTTP APIs.

## Motivation

### Problem Statement

Building execution environments for AI agents, code execution, or computational tasks typically involves:
- Complex setup and dependency management
- Security concerns with code execution
- Difficulty in scaling and deploying environments
- Lack of standardized interfaces between environments and clients of environments

### Goals

1. **Simplicity**: Simple APIs to interact with the environment from RL training code
2. **Type Safety**: Strongly-typed actions, observations, and state
3. **Isolation**: Each environment runs in its own Docker container
4. **Observability**: Leverage side-car container pattern to observe actions, observation tuples for an RL training eposide.


## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│             RL code(Client Application)                 │
│  ┌────────────────┐              ┌──────────────────┐   │
│  │  Environment   │              │  Environment     │   │
│  │  Client        │              │  Client          │   │
│  │ (HTTPEnvClient)│              │ (HTTPEnvClient)  │   │
│  └────────┬───────┘              └────────┬─────────┘   │
└───────────┼───────────────────────────────┼─────────────┘
            │ HTTP (reset, step, state)     │ HTTP
            │                               │
┌───────────▼───────────────────────────────▼─────────────┐
│              Docker Containers (Isolated)               │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ FastAPI Server       │    │ FastAPI Server       │   │
│  │   Environment        │    │   Environment        │   │
│  │   Logic              │    │   Logic              │   │
│  └──────────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Core Abstractions(Already available on the master)

#### 1. Environment (Server-Side)

```python
class Environment(ABC):
    """Base class for all environments."""

    @abstractmethod
    def reset(self) -> Observation:
        """Initialize new episode."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute action and return observation."""

    @property
    @abstractmethod
    def state(self) -> State:
        """Get current episode state."""
```

**Design Rationale**:
- Familiar interface for RL/environment practitioners
- Clear separation between action execution (step) and state management
- Abstract base class enforces contract across all environments

#### 2. HTTPEnvClient (Client-Side)

```python
class HTTPEnvClient(Generic[ActT, ObsT]):
    """Base class for HTTP environment clients."""

    def reset(self) -> StepResult[ObsT]:
        """Reset environment."""

    def step(self, action: ActT) -> StepResult[ObsT]:
        """Execute action."""

    def state(self) -> State:
        """Get current state."""

    def close(self) -> None:
        """Cleanup resources by signaling to the provider."""
```

**Design Rationale**:

The HTTPEnvClient serves as the primary interface for users to interact with environments, designed with several key principles:

- This base class handles all HTTP communication(resp, req) with the environment
- Generic types (`Generic[ActT, ObsT]`) provide compile-time type safety
- Each environment's concrete client class implements parsing step, observation, and state responses from the server into corresponding data models for the respective response.
- Example: `CodingEnv(HTTPEnvClient[CodeAction, CodeObservation])`
- `state()` method provides visibility into episode metadata
- Explicit `close()` ensures proper resource cleanup

#### 3. Container Providers

```python
class ContainerProvider(ABC):
    """Abstract base for container orchestration."""

    @abstractmethod
    def start_container(self, image: str, ...) -> str:
        """Start container and return base URL."""

    @abstractmethod
    def stop_container(self) -> None:
        """Stop and remove container."""

    @abstractmethod
    def wait_for_ready(self, base_url: str, timeout_s: float) -> None:
        """Wait for container to be ready."""
```

**Design Rationale**:
- Pluggable architecture supports multiple platforms (local Docker, K8s, other orchestration providers)
- Provider abstraction decouples client from deployment details and management with easy integration with existing orchestration solutions
- Consistent interface across all providers
- Higher level RL frameworks can implement their own container providers to integrate with their existing orchestration solutions.

### Key Design Decisions

In this RFC, we want to align on four decisions that will shape the overall design of the framework.

#### Decision 1: Baseline API Set

**Chosen Approach**: Define three core APIs as the baseline interface for this framework: `step`, `reset`, and `state`.

**Rationale**:
- **`reset()`**: Initializes a new episode and returns initial observation, providing a clean starting point for agent interactions
- **`step(action)`**: Executes an action and returns an observation, forming the core interaction loop
- **`state()`**: Provides visibility into the current episode state and metadata

These three APIs establish the minimum viable interface for environment interaction and are sufficient for basic RL training workflows. They align with established patterns from Gymnasium and similar frameworks, making them immediately familiar to practitioners.

**Scope**: This RFC focuses exclusively on these baseline APIs. Additional APIs (e.g., `render()`, `seed()`) will be explored in follow-up RFCs. The `actions()` method for action discovery is defined in RFC 004.

#### Decision 2: Environment-Computed Rewards

**Chosen Approach**: Rewards are computed inside the environment and returned as part of the observation.

**Rationale**:
- **Encapsulation**: Reward logic stays with the environment where domain knowledge resides
- **Consistency**: Ensures reward computation is deterministic and reproducible across different client implementations
- **Flexibility**: Environments can use internal state and context not visible to clients for reward computation
- **Standard Pattern**: Aligns with Gymnasium/Gym conventions where rewards are returned from `step()`

The `Observation` base class includes a `reward` field that environments populate:

```python
@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

This design enables environments to compute rewards based on:
- Action outcomes (e.g., exit codes, success/failure)
- Internal state transitions
- Multi-step trajectories
- Domain-specific metrics

Clients receive fully-formed observations with rewards already computed, simplifying the client-side RL loop.

#### Decision 3: HTTP-Based Communication

**Chosen Approach**: Use HTTP/REST for client-server communication

**Rationale**:
- HTTP based RPC is universal and well-understood than other alternatives like grpc or thrift
- Easy to debug with standard tools (curl, Postman)
- Supports language-agnostic clients
- FastAPI provides excellent developer experience

#### Decision 4: Docker-Based runtime isolation and packaging

**Chosen Approach**: Each environment runs in its own Docker container

**Rationale**:
- Strong isolation boundaries compared to process-based isolation
- Reproducible environments with packaged dependencies
- Easy dependency management via Dockerfile
- Industry-standard tooling


### Example Environments

**Purpose**: Test infrastructure, demonstrate patterns, verify deployments

#### Coding Environment

Executes Python code in a sandboxed environment:

```python
from envs.coding_env import CodeAction, CodingEnv

client = CodingEnv.from_docker_image("coding-env:latest")
result = client.step(CodeAction(code="print('Hello, World!')"))
print(result.observation.stdout)     # "Hello, World!\n"
print(result.observation.exit_code)  # 0
client.close()
```
