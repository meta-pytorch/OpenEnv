# envs/finqa_env/__init__.py
"""
FinQA Environment for OpenEnv.

A financial question-answering environment that evaluates LLMs on their ability
to answer complex financial questions using tool calls on SEC 10-K filing data.

Example:
    >>> from envs.finqa_env import FinQAEnv, FinQAAction, FinQAObservation
    >>>
    >>> # Start from Docker image
    >>> client = FinQAEnv.from_docker_image("finqa-env:latest")
    >>>
    >>> # Reset to get a question
    >>> result = client.reset()
    >>> print(f"Question: {result.observation.question}")
    >>> print(f"Company: {result.observation.company}")
    >>>
    >>> # Use tools to find the answer
    >>> result = client.step(FinQAAction(
    ...     tool_name="get_descriptions",
    ...     tool_args={"company_name": result.observation.company}
    ... ))
    >>>
    >>> # Submit final answer
    >>> result = client.step(FinQAAction(
    ...     tool_name="submit_answer",
    ...     tool_args={"answer": "6.118"}
    ... ))
    >>> print(f"Reward: {result.reward}")
    >>>
    >>> client.close()
"""

from .models import (
    FinQAAction,
    FinQAObservation,
    FinQAState,
    AVAILABLE_TOOLS,
)
from .client import FinQAEnv

__all__ = [
    "FinQAAction",
    "FinQAObservation",
    "FinQAState",
    "FinQAEnv",
    "AVAILABLE_TOOLS",
]
