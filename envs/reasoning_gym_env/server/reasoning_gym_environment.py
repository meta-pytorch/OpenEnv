# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reasoning Gym Environment Implementation.

A pure MCP environment that integrates with the Reasoning Gym library,
providing 100+ procedurally generated reasoning tasks with algorithmic verification.

All interactions happen through MCP tools:
- `get_question()`: Returns the current question and task type
- `submit_answer(answer)`: Submits answer, returns score and correct answer
- `get_task_info()`: Returns available tasks and configuration

Example:
    >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
    >>> env = ReasoningGymEnvironment()
    >>> env.reset(task_name="leg_counting", seed=42)
    >>>
    >>> # Get the question
    >>> obs = env.step(CallToolAction(tool_name="get_question", arguments={}))
    >>> print(obs.result)  # {"question": "How many legs do 2 dogs have?", "task": "leg_counting"}
    >>>
    >>> # Submit an answer
    >>> obs = env.step(CallToolAction(tool_name="submit_answer", arguments={"answer": "8"}))
    >>> print(obs.result)  # {"score": 1.0, "correct_answer": "8", "is_correct": True}
    >>>
    >>> # Switch tasks mid-session
    >>> env.reset(task_name="basic_arithmetic", task_config={"max_value": 50})
"""

from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


class ReasoningGymEnvironment(MCPEnvironment):
    """
    A pure MCP environment for reasoning tasks from Reasoning Gym.

    This environment exposes all functionality through MCP tools:
    - `get_question`: Returns the current question and task type
    - `submit_answer`: Submits an answer, returns score and correct answer
    - `get_task_info`: Returns available task configuration

    The environment inherits MCP support (ListToolsAction, CallToolAction)
    from the MCPEnvironment base class.

    Args:
        task_name: Single task type (default: "leg_counting")
        task_config: Task-specific configuration parameters
        task_specs: Composite dataset specs (mutually exclusive with task_name)
        dataset_size: Number of questions in the dataset
        seed: Random seed for reproducibility

    Example using MCPToolClient:
        >>> from openenv.core.mcp_client import MCPToolClient
        >>>
        >>> with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     env.reset(task_name="leg_counting", seed=42)
        ...     question = env.call_tool("get_question")
        ...     print(question["question"])
        ...     result = env.call_tool("submit_answer", answer="8")
        ...     print(f"Score: {result['score']}")
        ...     # Switch to different task
        ...     env.reset(task_name="basic_arithmetic")
    """

    def __init__(
        self,
        task_name: str = "leg_counting",
        task_config: Optional[dict] = None,
        task_specs: Optional[list] = None,
        dataset_size: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize the reasoning gym environment."""
        import reasoning_gym

        self._task_name = task_name
        self._task_config = task_config or {}
        self._task_specs = task_specs
        self._dataset_size = dataset_size
        self._seed = seed

        # Create the dataset
        if task_specs:
            self._dataset = reasoning_gym.create_dataset(
                "composite",
                size=dataset_size,
                seed=seed,
                datasets=task_specs,
            )
        else:
            self._dataset = reasoning_gym.create_dataset(
                task_name,
                size=dataset_size,
                seed=seed,
                **self._task_config,
            )

        self._entries = list(self._dataset)
        self._current_idx = 0
        self._current_entry: Optional[dict] = None
        self._done = False
        self._last_score = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Create MCP server and define tools inline
        mcp = FastMCP("reasoning_gym")

        # Store reference to self for closures
        env = self

        @mcp.tool
        def get_question() -> dict:
            """
            Get the current question.

            Returns:
                Dictionary with 'question' text and 'task' type.
                Never includes the answer.
            """
            if env._current_entry is None:
                return {"error": "No question available. Call reset() first."}

            task = env._current_entry.get("metadata", {}).get(
                "source_dataset", env._task_name
            )
            return {
                "question": env._current_entry["question"],
                "task": task,
            }

        @mcp.tool
        def submit_answer(answer: str) -> dict:
            """
            Submit an answer and get the score.

            Args:
                answer: The answer string to submit

            Returns:
                Dictionary with 'score' (0.0-1.0) and 'correct_answer'
            """
            if env._current_entry is None:
                return {"error": "No question available. Call reset() first."}

            if env._done:
                return {
                    "error": "Episode already complete. Call reset() for a new question."
                }

            score = env._dataset.score_answer(answer=answer, entry=env._current_entry)
            env._done = True
            env._last_score = score

            return {
                "score": score,
                "correct_answer": env._current_entry["answer"],
            }

        @mcp.tool
        def get_task_info() -> dict:
            """
            Get information about the current task configuration.

            Returns:
                Dictionary with task name, config, and dataset size
            """
            return {
                "task_name": env._task_name,
                "task_config": env._task_config,
                "dataset_size": env._dataset_size,
                "is_composite": env._task_specs is not None,
            }

        super().__init__(mcp)

    def _rebuild_dataset(self) -> None:
        """Rebuild the dataset with current configuration."""
        import reasoning_gym

        if self._task_specs:
            self._dataset = reasoning_gym.create_dataset(
                "composite",
                size=self._dataset_size,
                seed=self._seed,
                datasets=self._task_specs,
            )
        else:
            self._dataset = reasoning_gym.create_dataset(
                self._task_name,
                size=self._dataset_size,
                seed=self._seed,
                **self._task_config,
            )
        self._entries = list(self._dataset)
        self._current_idx = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        task_config: Optional[dict] = None,
        task_specs: Optional[list] = None,
        dataset_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment with a new question.

        Args:
            seed: Optional random seed for reseeding
            episode_id: Optional episode ID to use
            task_name: Optional task name to switch to (rebuilds dataset)
            task_config: Optional task configuration (rebuilds dataset)
            task_specs: Optional composite dataset specs (rebuilds dataset)
            dataset_size: Optional dataset size (rebuilds dataset)
            **kwargs: Additional reset options

        Returns:
            Observation indicating the environment is ready.
            Does NOT include the question or answer - use get_question tool.
        """
        needs_rebuild = False

        # Check if any configuration changed
        if task_name is not None and task_name != self._task_name:
            self._task_name = task_name
            self._task_specs = None  # task_name and task_specs are mutually exclusive
            needs_rebuild = True

        if task_config is not None and task_config != self._task_config:
            self._task_config = task_config
            needs_rebuild = True

        if task_specs is not None and task_specs != self._task_specs:
            self._task_specs = task_specs
            self._task_name = "composite"  # task_specs implies composite
            needs_rebuild = True

        if dataset_size is not None and dataset_size != self._dataset_size:
            self._dataset_size = dataset_size
            needs_rebuild = True

        if seed is not None and seed != self._seed:
            self._seed = seed
            needs_rebuild = True

        if needs_rebuild:
            self._rebuild_dataset()

        # Get next entry (cycle through dataset)
        self._current_entry = self._entries[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._entries)

        self._done = False
        self._last_score = 0.0
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Determine task name for metadata
        task = self._current_entry.get("metadata", {}).get(
            "source_dataset", self._task_name
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready", "task": task},
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.
        """
        return Observation(
            done=self._done,
            reward=self._last_score if self._done else 0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Updates done/reward state
        after submit_answer calls.

        Args:
            action: The MCP action to execute (ListToolsAction or CallToolAction)
            timeout_s: Optional timeout for the action
            **kwargs: Additional arguments

        Returns:
            Observation from the action execution
        """
        self._state.step_count += 1
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Update observation with done/reward state after tool calls
        if hasattr(obs, "done"):
            obs.done = self._done
        if hasattr(obs, "reward"):
            obs.reward = self._last_score if self._done else 0.0

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
