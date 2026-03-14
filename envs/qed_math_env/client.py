# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QED Math Environment Client.

Provides tool-calling style interactions with the QED Math environment
via MCP (Model Context Protocol).

Example:
    >>> with QEDMathEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     result = env.call_tool("get_problem")
    ...     result = env.call_tool("submit_proof", proof="By induction...")
"""

from typing import Any, Optional

from openenv.core.mcp_client import MCPToolClient


class QEDMathEnv(MCPToolClient):
    """
    Client for the QED Math Environment.

    Inherits MCP tool-calling interface from MCPToolClient:
    - ``list_tools()``: Discover available MCP tools
    - ``call_tool(name, **kwargs)``: Call a tool by name
    - ``reset(**kwargs)``: Reset the environment

    Example:
        >>> with QEDMathEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     result = env.call_tool("get_problem")
        ...     result = env.call_tool("submit_proof", proof="By induction...")
    """

    async def reset(self, problem_id: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Reset the environment, optionally selecting a specific problem.

        Args:
            problem_id: Optional problem identifier to load a specific problem.
                        If None, a problem is chosen randomly from the dataset.
            **kwargs: Additional reset parameters (e.g., seed).

        Returns:
            StepResult whose observation contains the initial ProblemObservation.
        """
        if problem_id is not None:
            kwargs["problem_id"] = problem_id
        return await super().reset(**kwargs)

    async def submit_proof(self, proof: str) -> Any:
        """
        Submit a proof attempt for the current problem.

        Args:
            proof: The proof text to submit for grading.

        Returns:
            Graded result dict with keys: score (0-7), feedback, reward (0.0-1.0).
        """
        return await self.call_tool("submit_proof", proof=proof)

    async def get_current_problem(self) -> Any:
        """
        Retrieve the current problem statement without resetting.

        Returns:
            Problem dict with keys: problem, reference_solution,
            grading_guidelines, problem_id, dataset_source.
        """
        return await self.call_tool("get_problem")

    async def get_grading_feedback(self) -> Any:
        """
        Retrieve the grading guidelines/rubric for the current problem.

        Returns:
            Grading guidelines string (Markdown rubric, 0-7 scale).
        """
        return await self.call_tool("get_grading_guidelines")
