# envs/finqa_env/server/finqa_environment.py
"""
FinQA Environment Implementation.

A financial question-answering environment that evaluates LLMs on their ability
to answer complex financial questions using tool calls on SEC 10-K filing data.
"""

import logging
import os
import random
import uuid
from typing import List, Dict, Any

import pandas as pd

from openenv.core.env_server import Environment
from ..models import FinQAAction, FinQAObservation, FinQAState, AVAILABLE_TOOLS
from .tools import FinQATools
from .rewards import compute_reward, extract_boxed_answer

logger = logging.getLogger(__name__)


class FinQAEnvironment(Environment):
    """
    Financial QA environment for RL training.

    Evaluates agents on their ability to answer financial questions by:
    - Exploring available tables for a company
    - Querying table metadata and executing SQL queries
    - Performing calculations
    - Submitting final answers

    Args:
        data_path: Path to the data directory containing benchmark_questions/ and input_companies/
        max_steps: Maximum number of tool calls per episode (default: 50)
        task: Task name - currently only 'finqa' supported (default: 'finqa')
    """

    def __init__(
        self,
        data_path: str = "./data",
        max_steps: int = 50,
        task: str = "finqa",
    ):
        super().__init__()

        self.data_path = data_path
        self.max_steps = max_steps
        self.task = task

        assert task == "finqa", "Only finqa task is supported"

        self.questions = self._load_questions()
        logger.info(f"Loaded {len(self.questions)} questions for task '{task}'")

        self.tools = FinQATools(data_path)

        # Shuffle dataset for sequential selection
        self._shuffled_questions = self.questions.copy()
        random.shuffle(self._shuffled_questions)
        self._question_index = 0

        self._state = FinQAState()
        self._history: List[Dict[str, Any]] = []

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load questions from the benchmark CSV."""
        csv_path = os.path.join(self.data_path, "benchmark_questions", f"{self.task}.csv")

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Benchmark file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        questions = []
        for _, row in df.iterrows():
            questions.append({
                "id": str(row.get("id", "")),
                "user_query": row["user_query"],
                "company": row["company"],
                "question": row["question"],
                "answer": row["answer"],
                "question_type": row.get("question_type", ""),
                "explanation": row.get("explanation", ""),
            })

        return questions

    def _get_next_question(self) -> Dict[str, Any]:
        """Get the next question using sequential shuffle selection."""
        if self._question_index >= len(self._shuffled_questions):
            # Reshuffle when exhausted
            random.shuffle(self._shuffled_questions)
            self._question_index = 0

        question = self._shuffled_questions[self._question_index]
        self._question_index += 1
        return question

    def reset(self) -> FinQAObservation:
        """
        Reset the environment for a new episode.

        Returns:
            Initial observation with the question
        """
        question = self._get_next_question()
        self._state = FinQAState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_question=question["user_query"],
            current_company=question["company"],
            ground_truth=question["answer"],
            question_id=question["id"],
        )
        self._history = []

        logger.info(f"Reset episode {self._state.episode_id} with question: {question['question'][:200]}...")

        return FinQAObservation(
            question=question["user_query"],
            company=question["company"],
            tool_result="",
            history=[],
            step_count=0,
            available_tools=AVAILABLE_TOOLS.copy(),
            done=False,
            reward=None,
        )

    def step(self, action: FinQAAction) -> FinQAObservation:
        """
        Execute a tool call and return the result.

        Args:
            action: The tool call to execute

        Returns:
            Observation with tool result
        """
        self._state.step_count += 1

        if action.tool_name not in AVAILABLE_TOOLS:
            tool_result = f"Error: Unknown tool '{action.tool_name}'. Available: {AVAILABLE_TOOLS}"
            is_final = False
        else:
            tool_result, is_final = self.tools.execute_tool(
                action.tool_name, action.tool_args
            )

        self._history.append({
            "step": self._state.step_count,
            "tool": action.tool_name,
            "args": action.tool_args,
            "result": tool_result,
        })

        done = False
        reward = None

        if is_final:
            done = True
            submitted_answer = action.tool_args.get("answer", "")
            reward = compute_reward(submitted_answer, self._state.ground_truth)
            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"submitted='{submitted_answer}', truth='{self._state.ground_truth}', reward={reward}"
            )

        elif self._state.step_count >= self.max_steps:
            done = True
            reward = 0.0
            tool_result = f"Max steps ({self.max_steps}) reached without submitting answer. Episode terminated."
            logger.info(f"Episode {self._state.episode_id} terminated: max steps reached")

        return FinQAObservation(
            question=self._state.current_question,
            company=self._state.current_company,
            tool_result=tool_result,
            history=self._history.copy(),
            step_count=self._state.step_count,
            available_tools=AVAILABLE_TOOLS.copy(),
            done=done,
            reward=reward,
            metadata={
                "ground_truth": self._state.ground_truth if done else None,
            },
        )

    @property
    def state(self) -> FinQAState:
        """Get the current environment state."""
        return self._state
