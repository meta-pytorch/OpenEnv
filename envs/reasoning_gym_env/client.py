# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reasoning Gym Environment Client.

This module provides the client for connecting to a Reasoning Gym Environment server.
ReasoningGymEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    ...     # Configure task at reset time
    ...     env.reset(task_name="leg_counting", seed=42)
    ...
    ...     # Get the question
    ...     question = env.call_tool("get_question")
    ...     print(question["question"])
    ...
    ...     # Submit an answer
    ...     result = env.call_tool("submit_answer", answer="8")
    ...     print(f"Score: {result['score']}")
    ...
    ...     # Switch to different task mid-session
    ...     env.reset(task_name="basic_arithmetic", task_config={"max_value": 100})
"""

from openenv.core.mcp_client import MCPToolClient


class ReasoningGymEnv(MCPToolClient):
    """
    Client for the Reasoning Gym Environment.

    This client provides a simple interface for interacting with the Reasoning Gym
    Environment via MCP tools. It inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Reset parameters for task configuration:
        - task_name: Task type (e.g., "leg_counting", "basic_arithmetic")
        - task_config: Task-specific settings (e.g., {"max_value": 100})
        - task_specs: Composite dataset specs (for multi-task training)
        - dataset_size: Number of questions in the dataset
        - seed: Random seed for reproducibility

    Example:
        >>> with ReasoningGymEnv(base_url="http://localhost:8000") as env:
        ...     # Configure task at reset time
        ...     env.reset(task_name="leg_counting", seed=42)
        ...
        ...     # Get question
        ...     question = env.call_tool("get_question")
        ...     print(question["question"])
        ...
        ...     # Submit answer
        ...     result = env.call_tool("submit_answer", answer="42")
        ...     print(f"Score: {result['score']}")
        ...
        ...     # Switch tasks mid-session
        ...     env.reset(task_name="basic_arithmetic", task_config={"max_value": 50})

    Example with Docker:
        >>> env = ReasoningGymEnv.from_docker_image("reasoning-gym-env:latest")
        >>> try:
        ...     env.reset(task_name="leg_counting", seed=42)
        ...     question = env.call_tool("get_question")
        ...     result = env.call_tool("submit_answer", answer="8")
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
