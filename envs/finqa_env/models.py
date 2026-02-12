# envs/finqa_env/models.py
"""
Action/Observation/State types for the FinQA environment.

FinQA is a financial question-answering benchmark that evaluates LLMs on their
ability to answer complex financial questions using tool calls (SQL queries,
calculations, etc.) on SEC 10-K filing data.
"""

from typing import Dict, List, Any

from pydantic import Field

from openenv.core.env_server import Action, Observation, State


# Tool names - defined statically to avoid circular imports
AVAILABLE_TOOLS = ["get_descriptions", "get_table_info", "sql_query", "submit_answer"]


class FinQAAction(Action):
    """
    Action taken by the agent - a tool call.

    Attributes:
        tool_name: One of: get_descriptions, get_table_info, sql_query, submit_answer
        tool_args: Arguments for the tool call
    """
    tool_name: str
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class FinQAObservation(Observation):
    """
    Observation returned after each step.

    Attributes:
        question: The financial question being asked
        company: The company the question is about
        tool_result: Result of the last tool call (empty string on reset)
        history: List of previous tool calls and their results
        step_count: Current step number in the episode
        available_tools: List of tools the agent can use
        # Inherited from Observation: done, reward, metadata
    """
    question: str = ""
    company: str = ""
    tool_result: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    available_tools: List[str] = Field(default_factory=lambda: AVAILABLE_TOOLS.copy())


class FinQAState(State):
    """
    Internal environment state for tracking the current episode.

    All fields are set during reset() and are essential for episode tracking.

    Attributes:
        current_question: The question being asked
        current_company: The company the question is about
        ground_truth: The expected answer for reward computation
        question_id: Identifier for the current question
        # Inherited from State: episode_id, step_count
    """
    current_question: str = ""
    current_company: str = ""
    ground_truth: str = ""
    question_id: str = ""
