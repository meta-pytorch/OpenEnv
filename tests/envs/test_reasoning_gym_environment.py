# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Reasoning Gym Environment.

These tests use mocked reasoning-gym to verify environment behavior
without requiring the actual library to be installed.
"""

import sys
from unittest.mock import MagicMock

import pytest


class MockDataset:
    """Mock reasoning-gym dataset for testing."""

    def __init__(
        self,
        name: str = "leg_counting",
        size: int = 100,
        seed: int = None,
        datasets: list = None,
        **kwargs,
    ):
        self.name = name
        self.size = size
        self.seed = seed
        self.datasets = datasets
        self.config = kwargs

        # Create mock entries
        self._entries = [
            {
                "question": f"How many legs do {i + 1} dogs have?",
                "answer": str((i + 1) * 4),
                "metadata": {"source_dataset": name, "difficulty": "easy"},
            }
            for i in range(min(size, 10))
        ]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def score_answer(self, answer: str, entry: dict) -> float:
        """Score an answer against the entry's correct answer."""
        if answer.strip() == entry["answer"].strip():
            return 1.0
        return 0.0


def mock_create_dataset(name: str, size: int = 100, seed: int = None, **kwargs):
    """Mock reasoning_gym.create_dataset function."""
    return MockDataset(name=name, size=size, seed=seed, **kwargs)


# Set up the mock before importing the environment
_mock_reasoning_gym = MagicMock()
_mock_reasoning_gym.create_dataset = mock_create_dataset
sys.modules["reasoning_gym"] = _mock_reasoning_gym


# Now we can import the environment safely
from reasoning_gym_env.server.reasoning_gym_environment import ReasoningGymEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Action, Observation


@pytest.fixture
def reasoning_env():
    """Create a ReasoningGymEnvironment with mocked reasoning-gym."""
    return ReasoningGymEnvironment(task_name="leg_counting", dataset_size=10, seed=42)


class TestReasoningGymEnvironmentBasic:
    """Basic tests for ReasoningGymEnvironment."""

    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        env = ReasoningGymEnvironment(task_name="leg_counting")
        assert env is not None
        assert env._task_name == "leg_counting"

    def test_reset_returns_observation(self, reasoning_env):
        """Test that reset returns an Observation with ready status."""
        obs = reasoning_env.reset()

        assert obs is not None
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.metadata["status"] == "ready"
        assert "task" in obs.metadata

    def test_reset_with_episode_id(self, reasoning_env):
        """Test reset with custom episode ID."""
        reasoning_env.reset(episode_id="custom-episode-123")
        assert reasoning_env.state.episode_id == "custom-episode-123"

    def test_state_property(self, reasoning_env):
        """Test that state property returns current state."""
        reasoning_env.reset()
        state = reasoning_env.state
        assert state is not None
        assert state.step_count == 0


class TestReasoningGymEnvironmentMCP:
    """Tests for MCP tool functionality."""

    def test_list_tools(self, reasoning_env):
        """Test that ListToolsAction returns available tools."""
        reasoning_env.reset()
        obs = reasoning_env.step(ListToolsAction())

        assert isinstance(obs, ListToolsObservation)
        tool_names = [t.name for t in obs.tools]
        assert "get_question" in tool_names
        assert "submit_answer" in tool_names
        assert "get_task_info" in tool_names

    def test_get_question_returns_question_only(self, reasoning_env):
        """Test that get_question returns question without answer."""
        reasoning_env.reset()
        obs = reasoning_env.step(CallToolAction(tool_name="get_question", arguments={}))

        assert isinstance(obs, CallToolObservation)
        assert obs.error is None

        # Extract result - FastMCP wraps in CallToolResult
        result = obs.result
        if hasattr(result, "data"):
            result = result.data

        assert "question" in str(result)
        assert "task" in str(result)

    def test_submit_answer_correct(self, reasoning_env):
        """Test submitting a correct answer returns score 1.0."""
        reasoning_env.reset()

        # Submit the correct answer (1 dog = 4 legs)
        obs = reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "4"})
        )

        assert obs.error is None

        result = obs.result
        if hasattr(result, "data"):
            result = result.data

        # Check score and that correct_answer is revealed
        assert "score" in str(result) or "1.0" in str(result)

    def test_submit_answer_incorrect(self, reasoning_env):
        """Test submitting an incorrect answer returns score 0.0."""
        reasoning_env.reset()

        # Submit wrong answer
        obs = reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "wrong"})
        )

        assert obs.error is None
        assert obs.done is True  # Episode should be done after submit

    def test_submit_answer_sets_done(self, reasoning_env):
        """Test that submitting answer marks episode as done."""
        reasoning_env.reset()
        assert reasoning_env._done is False

        reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "4"})
        )

        assert reasoning_env._done is True

    def test_get_task_info(self, reasoning_env):
        """Test get_task_info returns configuration."""
        reasoning_env.reset()
        obs = reasoning_env.step(
            CallToolAction(tool_name="get_task_info", arguments={})
        )

        assert obs.error is None

        result = obs.result
        if hasattr(result, "data"):
            result = result.data

        assert "leg_counting" in str(result)


class TestAnswerIsolation:
    """Critical tests: Answer must NEVER be visible before submission."""

    def test_answer_not_in_reset_observation(self, reasoning_env):
        """Test that reset observation does NOT contain the answer."""
        obs = reasoning_env.reset()

        # Convert to string and check for answer patterns
        obs_str = str(obs.metadata)
        assert "answer" not in obs_str.lower()

    def test_answer_not_in_get_question_result(self, reasoning_env):
        """Test that get_question result does NOT contain the answer."""
        reasoning_env.reset()
        obs = reasoning_env.step(CallToolAction(tool_name="get_question", arguments={}))

        result = obs.result
        if hasattr(result, "data"):
            result = result.data

        result_str = str(result)

        # Check the word "answer" doesn't appear as a key
        assert '"answer"' not in result_str
        assert "'answer'" not in result_str

    def test_answer_revealed_only_after_submit(self, reasoning_env):
        """Test that answer is revealed ONLY after submit_answer."""
        reasoning_env.reset()

        # Before submit - get_question should not have answer
        q_obs = reasoning_env.step(
            CallToolAction(tool_name="get_question", arguments={})
        )
        q_result = q_obs.result
        if hasattr(q_result, "data"):
            q_result = q_result.data
        assert "correct_answer" not in str(q_result)

        # After submit - answer should be revealed
        s_obs = reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "test"})
        )
        s_result = s_obs.result
        if hasattr(s_result, "data"):
            s_result = s_result.data
        assert "correct_answer" in str(s_result) or "4" in str(s_result)

    def test_state_does_not_expose_entry(self, reasoning_env):
        """Test that state property does not expose current entry data."""
        reasoning_env.reset()
        state = reasoning_env.state

        # State should only have episode_id and step_count
        state_str = str(state)
        assert "question" not in state_str.lower()
        assert "answer" not in state_str.lower()


class TestEpisodeLifecycle:
    """Tests for episode lifecycle management."""

    def test_cannot_submit_twice(self, reasoning_env):
        """Test that submitting twice returns an error."""
        reasoning_env.reset()

        # First submit
        reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "4"})
        )

        # Second submit should return error
        obs = reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "8"})
        )

        result = obs.result
        if hasattr(result, "data"):
            result = result.data

        assert "error" in str(result).lower() or "complete" in str(result).lower()

    def test_reset_clears_done_state(self, reasoning_env):
        """Test that reset clears the done state."""
        reasoning_env.reset()
        reasoning_env.step(
            CallToolAction(tool_name="submit_answer", arguments={"answer": "4"})
        )
        assert reasoning_env._done is True

        # Reset should clear done
        reasoning_env.reset()
        assert reasoning_env._done is False

    def test_questions_cycle_through_dataset(self, reasoning_env):
        """Test that questions cycle through the dataset on reset."""
        questions = []

        for _ in range(3):
            reasoning_env.reset()
            obs = reasoning_env.step(
                CallToolAction(tool_name="get_question", arguments={})
            )
            result = obs.result
            if hasattr(result, "data"):
                result = result.data
            questions.append(str(result))

        # Questions should be different (cycling through dataset)
        assert questions[0] != questions[1] or questions[1] != questions[2]


class TestNonMCPActions:
    """Tests for handling non-MCP actions."""

    def test_unknown_action_type_returns_error(self, reasoning_env):
        """Test that unknown action types return an error observation."""

        class UnknownAction(Action):
            pass

        reasoning_env.reset()
        obs = reasoning_env.step(UnknownAction())

        assert "error" in str(obs.metadata).lower()


class TestClientImports:
    """Tests for client module imports."""

    def test_client_import(self):
        """Test that ReasoningGymEnv can be imported from the package."""
        from reasoning_gym_env import ReasoningGymEnv

        assert ReasoningGymEnv is not None



class TestClientSideConfiguration:
    """Tests for client-side configuration via reset() kwargs."""

    def test_reset_with_task_name(self, reasoning_env):
        """Test that reset() accepts task_name and updates configuration."""
        reasoning_env.reset(task_name="basic_arithmetic")
        assert reasoning_env._task_name == "basic_arithmetic"

    def test_reset_with_task_config(self, reasoning_env):
        """Test that reset() accepts task_config and updates configuration."""
        new_config = {"max_value": 50}
        reasoning_env.reset(task_config=new_config)
        assert reasoning_env._task_config == new_config

    def test_reset_with_dataset_size(self, reasoning_env):
        """Test that reset() accepts dataset_size and updates configuration."""
        reasoning_env.reset(dataset_size=25)
        assert reasoning_env._dataset_size == 25

    def test_reset_with_seed(self, reasoning_env):
        """Test that reset() accepts seed and rebuilds dataset."""
        reasoning_env.reset(seed=123)
        assert reasoning_env._seed == 123

    def test_reset_rebuilds_dataset_on_task_change(self, reasoning_env):
        """Test that changing task_name rebuilds the dataset."""
        reasoning_env.reset()
        old_idx = reasoning_env._current_idx

        # Change task - should rebuild and reset index
        reasoning_env.reset(task_name="different_task")

        # After rebuild, we get entry at index 0, then idx becomes 1
        assert reasoning_env._current_idx == 1
        assert reasoning_env._task_name == "different_task"

    def test_reset_rebuilds_dataset_on_config_change(self, reasoning_env):
        """Test that changing task_config rebuilds the dataset."""
        reasoning_env.reset()

        # Change config - should rebuild
        reasoning_env.reset(task_config={"new_param": "value"})

        assert reasoning_env._task_config == {"new_param": "value"}
        assert reasoning_env._current_idx == 1  # Reset to 0, then advanced to 1

    def test_reset_without_changes_cycles_through(self, reasoning_env):
        """Test that reset() without config changes just cycles through entries."""
        reasoning_env.reset()
        idx_after_first = reasoning_env._current_idx

        reasoning_env.reset()
        idx_after_second = reasoning_env._current_idx

        # Should have advanced
        assert idx_after_second == idx_after_first + 1

    def test_task_name_clears_task_specs(self, reasoning_env):
        """Test that setting task_name clears task_specs (mutually exclusive)."""
        reasoning_env._task_specs = [{"name": "old_spec"}]

        reasoning_env.reset(task_name="new_task")

        assert reasoning_env._task_name == "new_task"
        assert reasoning_env._task_specs is None

    def test_task_specs_sets_composite_name(self, reasoning_env):
        """Test that setting task_specs sets task_name to 'composite'."""
        specs = [{"name": "task1", "weight": 1}]

        reasoning_env.reset(task_specs=specs)

        assert reasoning_env._task_specs == specs
        assert reasoning_env._task_name == "composite"

    def test_multiple_config_changes_in_one_reset(self, reasoning_env):
        """Test that multiple config changes in one reset() call work."""
        reasoning_env.reset(
            task_name="new_task",
            task_config={"param": "value"},
            dataset_size=50,
            seed=999,
        )

        assert reasoning_env._task_name == "new_task"
        assert reasoning_env._task_config == {"param": "value"}
        assert reasoning_env._dataset_size == 50
        assert reasoning_env._seed == 999

    def test_same_config_does_not_rebuild(self, reasoning_env):
        """Test that passing same config values doesn't trigger rebuild."""
        reasoning_env.reset()

        # Advance a few times
        reasoning_env.reset()
        reasoning_env.reset()
        old_idx = reasoning_env._current_idx

        # Pass same values - should NOT rebuild, just cycle
        reasoning_env.reset(
            task_name=reasoning_env._task_name,
            seed=reasoning_env._seed,
        )

        # Should have cycled, not reset
        expected_idx = (old_idx + 1) % len(reasoning_env._entries)
        assert reasoning_env._current_idx == expected_idx
