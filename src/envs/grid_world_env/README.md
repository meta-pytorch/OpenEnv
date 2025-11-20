# Grid World Environment

This directory contains the implementation of a simple 5x5 Grid World environment, designed to serve two primary purposes within the OpenEnv ecosystem:

1.  **A basic Reinforcement Learning (RL) testbed:** Providing a straightforward, deterministic environment for quick prototyping and testing of RL agents.
2.  **A detailed "How-To" guide for building new OpenEnv environments:** Demonstrating the architectural patterns, best practices, and core components required to integrate a custom environment into the OpenEnv framework.

---

## 🚀 Environment Overview

The Grid World environment features:

* **Grid Size:** A 5x5 square grid.
* **Agent:** Starts at position `(0,0)` (top-left).
* **Goal:** Fixed at `(4,4)` (bottom-right).
* **Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`.
* **Dynamics:** Deterministic. An action always moves the agent one step in the chosen direction, unless it would move off the grid, in which case the agent stays in its current cell (e.g., trying `UP` from `(0,0)` keeps the agent at `(0,0)`).
* **Reward Function (Sparse):**
    * `-0.1` for every step taken (a "living cost" or "step penalty").
    * `+1.0` for reaching the goal at `(4,4)`. This also terminates the episode.
* **Episode Termination:** The episode ends when the agent reaches the goal.

### Example Gameplay

Imagine the agent trying to find the goal:

Reset: Agent at (0,0) -> Obs(x=0, y=0, reward=None, done=False)
Agent moves to (1,0)->Obs(x=1, y=0, reward=-0.1, done=False)
Step Right: Agent moves to (1,1)-> Obs(x=1,y=1, reward=-0.1,done=False)
....
Step Right(from 4,3):Agent moves to (4,4)->Obs(x=4,y=4,reward=1.0, done=True)


---

## 🛠️ How to Build an OpenEnv Environment: A Detailed Guide

This section explains the structure and key design choices of the Grid World environment, serving as a template for new OpenEnv contributions.

### 1. Scaffolding the Environment Folder

The 'grid_world_env' directory structure was initially generated using the OpenEnv CLI tool. This provides a foundational template that adheres to the expected OpenEnv architecture.
To create a new environment folder like 'grid_world_env', you would typically run a command: 
```bash
# Example command to scaffold a new environment
Open init grid_world_env
```

### Directory Structure
src/envs/grid_world_env
-server
   __init__.py.              # Package initializer for the server side 
   app.py                    # The FastAPI application entry point
   Dockerfile                # Instruction for building the Docker container for the environment server
   grid_world_environment.py # The core environmentlogic 
   requirement.txt           # Python Dependencies specific to the environment server
__init__.py                  # Package initializer for the client-side
client.py                    # Defines the python client for interacting with the environment HTTP server.
models.py                    # Defines the data structures (Action, Observation, State) for the environment.
openenv.yaml                 # OpenEnv metadata
README.md
test_grid_world.sh           # A shell script to build, run, and test the environment container

### Core Components Explained

#### 1. `models.py` (The Data Contract)

This file is paramount. It defines the "language" through which the client (your RL agent) and the server (your environment logic) communicate.

* **`from __future__ import annotations`**: Essential for type hinting in Python, especially for forward references.
* **`@dataclass`**: Used for `GridWorldAction`, `GridWorldObservation`, and `GridWorldState`.
    * **Purpose:** Dataclasses automatically generate standard methods like `__init__`, `__repr__`, and `__eq__`, making data structures clean and readable. Crucially, they are easily **serializable** (Python object to JSON/bytes) and **deserializable** (JSON/bytes to Python object) by FastAPI, which is how data travels over HTTP.
* **`Enum` for `MoveAction`**: Provides a type-safe way to define allowed actions (`UP`, `DOWN`, `LEFT`, `RIGHT`). This prevents typos and ensures only valid actions are sent to the environment.
* **Inheriting from `Action`, `Observation`, `State`**:
    * `GridWorldAction(Action)`: Ensures your action model conforms to the OpenEnv API's expected action structure.
    * `GridWorldObservation(Observation)`: Defines what information the agent *sees* after taking a step (its position, message, reward, done status).
    * `GridWorldState(State)`: Represents the *entire internal state* of the environment. This is typically used for logging, debugging, or full state-based algorithms, but usually *not* directly observed by the agent during training unless the problem dictates it. It includes hidden details like `goal_x`, `goal_y`.

#### 2. `server/grid_world_environment.py` (The Environment Logic)

This file contains the "brain" and "physics engine" of your environment.

* **`class GridWorldEnvironment(Environment):`**: Your environment must inherit from `core.env_server.Environment`. This is the fundamental interface for OpenEnv environments.
* **`__init__(self, *args, **kwargs):`**:
    * Initializes the environment's fixed properties (`grid_size`, `goal_pos`).
    * **`self._state = GridWorldState(...)`**: **Crucial for persistence.** This creates a single, persistent `GridWorldState` object (`_state`). All environment updates modify this *same object*. This ensures the environment "remembers" the agent's position, step count, etc., across `step()` calls, mirroring the `atari_env` pattern.
* **`reset(self) -> GridWorldObservation:`**:
    * **Purpose:** Prepares the environment for a new episode.
    * Resets the agent's position (`agent_x=0`, `agent_y=0`).
    * Resets `episode_steps` and `step_count`.
    * **`self._state.episode_id = str(uuid.uuid4())`**: Assigns a unique identifier to each episode, valuable for logging and reproducibility.
    * Returns the initial `GridWorldObservation` for the agent to perceive at the start of the episode.
* **`step(self, action: GridWorldAction) -> GridWorldObservation:`**:
    * **Purpose:** Advances the environment by one timestep based on the agent's action.
    * **`self._state.step_count += 1`**: Increments the standard OpenEnv step counter.
    * **Action Interpretation:** Decodes the `GridWorldAction` (e.g., `MoveAction.DOWN`).
    * **Physics/Dynamics:** Updates `current_x`, `current_y` based on the action.
    * **Boundary Conditions:** `max(0, min(current_x, self.grid_size - 1))` prevents the agent from moving off the grid.
    * **State Update:** Modifies `self._state.agent_x` and `self._state.agent_y` to reflect the new position.
    * **Reward Function:** Calculates the `reward` (e.g., `-0.1` step penalty, `+1.0` for goal) and `done` status. This is the **teacher's feedback**.
    * Returns the new `GridWorldObservation`, `reward`, and `done` status to the agent.
* **`@property def state(self) -> GridWorldState:`**:
    * **Purpose:** Provides read-only access to the *full internal state* of the environment.
    * Allows external tools (like loggers or advanced debugging) to inspect all variables, including those not exposed in the `Observation` (e.g., `goal_x`, `goal_y`).

#### 3. `server/app.py` (The Server Application)

This file transforms your Python `GridWorldEnvironment` into a web API.

* **`from core.env_server import create_fastapi_app`**: Leverages the OpenEnv core utility to generate the FastAPI application.
* **`env = GridWorldEnvironment()`**: Creates a single instance of your environment. This instance lives for the entire duration of the server's operation.
* **`app = create_fastapi_app(env, GridWorldAction, GridWorldObservation)`**: This line automatically sets up the HTTP endpoints (`/reset`, `/step`, `/state`, `/health`) based on your `env` object and the defined `Action` and `Observation` models. FastAPI handles all the JSON serialization and deserialization using your `@dataclass` models.

#### 4. `server/Dockerfile` (The Containerization Blueprint)

This file provides instructions for Docker to build a self-contained, runnable package of your environment.

* **`FROM ${BASE_IMAGE}`**: Starts with `envtorch-base:latest`, providing a consistent base operating system and core Python setup.
* **`COPY requirements.txt /app/requirements.txt`**: Copies your server's Python dependencies.
* **`RUN pip install ...`**: Installs `fastapi` and `uvicorn`, which are crucial for running the web server.
* **`COPY src/core/ /app/src/core/`**: Ensures the core OpenEnv server utilities are available inside the container.
* **`COPY src/envs/grid_world_env/ /app/src/envs/grid_world_env/`**: Copies your environment's code into the container.
* **`EXPOSE 8000`**: Informs Docker that the server will listen on port 8000.
* **`HEALTHCHECK`**: Defines how Docker can verify if your server is running correctly (by checking the `/health` endpoint).
* **`CMD ["uvicorn", ...]`**: The command that Docker executes when the container starts, launching your FastAPI application using Uvicorn.

#### 5. `client.py` (The Python Client)

This file provides a convenient Python interface for your RL agent to interact with the Dockerized environment.

* **`class GridWorldEnv(HTTPEnvClient[GridWorldAction, GridWorldObservation]):`**:
    * Inherits from `core.http_env_client.HTTPEnvClient`, which provides the base logic for sending HTTP requests (`/reset`, `/step`, `/state`).
    * The generic types `[GridWorldAction, GridWorldObservation]` are crucial for **type safety** and allow the `HTTPEnvClient` to correctly handle the conversion of Python objects to/from JSON.
* **`def __init__(self, *args, **kwargs):`**:
    * This explicit `__init__` sets the `action_model`, `observation_model`, and `state_model` for the base `HTTPEnvClient`. This is essential so the client knows how to serialize your actions and deserialize the server's observations and state.
* **`step_move(self, move: MoveAction) -> StepResult[GridWorldObservation]:`**:
    * A helper method specific to Grid World. It simplifies making moves by allowing `env.step_move(MoveAction.RIGHT)` instead of manually creating a `GridWorldAction` object. This improves usability for the end-user.

#### 6. `test_grid_world.sh` (The Integration Test Script)

This shell script automates the entire process of building, running, and verifying your environment.

* **`set -e`**: Ensures the script exits immediately if any command fails.
* **`docker build ...`**: Builds your Docker image. The `--no-cache` flag (used during development) forces a fresh build, ensuring code changes are picked up.
* **`docker run -d -p 8000:8000 ...`**: Starts your environment in a Docker container, mapping its internal port 8000 to your machine's port 8000.
* **`curl ...`**: Makes HTTP requests to your running environment's endpoints (`/health`, `/reset`, `/step`, `/state`).
* **`jq -e ...`**: Parses JSON responses to verify that the environment behaves as expected.
* **`docker stop ...; docker rm ...`**: Cleans up the container after tests, leaving a tidy system.

---

## 🚀 Getting Started

To run and test the Grid World environment locally:

1.  Navigate to the root of the `OpenEnv` repository.
2.  Run the test script:
    ```bash
    ./src/envs/grid_world_env/test_grid_world.sh
    ```
    This script will build the Docker image, start the container, run a series of `reset`, `step`, and `state` tests, and then clean up.

---

## Conclusion

This Grid World environment serves as a robust example for building custom environments within the OpenEnv framework. By following these patterns, developers can integrate their own simulations, games, or control problems, leveraging Docker for consistent deployment and FastAPI for a flexible API interface.

---

