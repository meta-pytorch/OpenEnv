# src/envs/finqa_env/server/test_finqa_env.py
"""
Unit tests for the FinQA environment.

Run from OpenEnv repo root:
    python -m pytest src/envs/finqa_env/server/test_finqa_env.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import pytest
from envs.finqa_env.models import FinQAAction, FinQAObservation, FinQAState, AVAILABLE_TOOLS
from envs.finqa_env.server.finqa_environment import FinQAEnvironment
from envs.finqa_env.server.rewards import compute_reward, parse_number, extract_boxed_answer
from envs.finqa_env.server.tools import FinQATools


# Use the symlinked data path
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")


class TestRewards:
    """Test reward computation logic."""

    def test_exact_match(self):
        assert compute_reward("6.118", "6.118") == 1.0

    def test_boxed_format(self):
        assert compute_reward("6.118", r"\boxed{6.118}") == 1.0
        assert compute_reward(r"\boxed{6.118}", "6.118") == 1.0

    def test_tolerance(self):
        # Within 1% tolerance
        assert compute_reward("6.12", "6.118") == 1.0
        assert compute_reward("6.1", "6.118") == 1.0

    def test_incorrect(self):
        assert compute_reward("5.0", "6.118") == 0.0
        assert compute_reward("100", "6.118") == 0.0

    def test_parse_number(self):
        assert parse_number("6.118") == 6.118
        assert parse_number("1,234.56") == 1234.56
        assert parse_number("20%") == 0.2
        assert parse_number("1/2") == 0.5

    def test_extract_boxed(self):
        assert extract_boxed_answer(r"\boxed{6.118}") == "6.118"
        assert extract_boxed_answer("no boxed here") is None


class TestTools:
    """Test tool implementations."""

    @pytest.fixture
    def tools(self):
        return FinQATools(DATA_PATH)

    def test_get_available_companies(self, tools):
        companies = tools.get_available_companies()
        assert len(companies) > 0
        assert "alphabet" in companies

    def test_get_descriptions(self, tools):
        result = tools.get_descriptions("alphabet")
        assert "Error" not in result
        assert "us_gaap_" in result  # Should contain GAAP table names

    def test_get_descriptions_invalid_company(self, tools):
        result = tools.get_descriptions("nonexistent_company")
        assert "Error" in result

    def test_get_table_info(self, tools):
        result = tools.get_table_info(
            "alphabet",
            "us_gaap_ScheduleOfIncomeBeforeIncomeTaxDomesticAndForeignTableTextBlock"
        )
        assert "Error" not in result
        assert "column_dtypes" in result

    def test_sql_query_no_filter(self, tools):
        result = tools.sql_query("alphabet", "some_table", "SELECT * FROM some_table")
        assert "Error" in result

class TestEnvironment:
    """Test environment logic."""

    @pytest.fixture
    def env(self):
        return FinQAEnvironment(data_path=DATA_PATH, max_steps=10)

    def test_reset(self, env):
        obs = env.reset()
        assert isinstance(obs, FinQAObservation)
        assert obs.question != ""
        assert obs.company != ""
        assert obs.step_count == 0
        assert obs.done is False
        assert obs.reward is None

    def test_step_get_descriptions(self, env):
        obs = env.reset()
        action = FinQAAction(
            tool_name="get_descriptions",
            tool_args={"company_name": obs.company}
        )
        obs = env.step(action)
        assert obs.step_count == 1
        assert obs.tool_result != ""
        assert "Error" not in obs.tool_result or "not found" in obs.tool_result.lower()

    def test_step_submit_answer(self, env):
        env.reset()
        action = FinQAAction(
            tool_name="submit_answer",
            tool_args={"answer": "6.118"}
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward is not None
        assert obs.reward in [0.0, 1.0]

    def test_max_steps_termination(self, env):
        env.reset()
        for _ in range(10):
            action = FinQAAction(
                tool_name="get_descriptions",
                tool_args={"company_name": "test"}
            )
            obs = env.step(action)
            if obs.done:
                break

        assert obs.done is True
        assert obs.reward == 0.0  # No answer submitted

    def test_state_property(self, env):
        env.reset()
        state = env.state
        assert isinstance(state, FinQAState)
        assert state.episode_id is not None
        assert state.current_question is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
