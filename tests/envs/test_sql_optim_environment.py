# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the SQL Query Optimization environment.

Covers the server-side environment (reset/step/state driven against real
DuckDB execution), the wire models, and the client's payload/parse helpers.
The client's transport is exercised separately by the WebSocket integration
tests; here we unit-test the pure parsing methods without a live server.
"""

import os
import sys

import pytest

# Add the repo root so `envs.*` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from envs.sql_optim_env import (  # noqa: E402
    SQLOptimAction,
    SQLOptimEnv,
    SQLOptimObservation,
    SQLOptimState,
)
from envs.sql_optim_env.server.sql_optim_environment import (  # noqa: E402
    DEFAULT_TASK_ID,
    SQLOptimEnvironment,
)
from envs.sql_optim_env.server.tasks import TASKS  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """A single environment instance (DuckDB tables are built once)."""
    return SQLOptimEnvironment()


@pytest.fixture(scope="module")
def executor():
    """The shared DuckDB query executor (tables built once)."""
    from envs.sql_optim_env.server.executor import get_executor

    return get_executor()


def _good_action() -> SQLOptimAction:
    """A result-preserving task-1 rewrite: sargable predicates, projection kept."""
    return SQLOptimAction(
        suggestions=[
            {
                "issue_type": "select_star",
                "line": 1,
                "description": "SELECT * is wasteful",
                "severity": "high",
                "fix": "project columns",
            },
            {
                "issue_type": "non_sargable_cast",
                "line": 3,
                "description": "CAST blocks pruning",
                "severity": "high",
                "fix": "compare int",
            },
            {
                "issue_type": "function_on_date_column",
                "line": 4,
                "description": "year() non-sargable",
                "severity": "medium",
                "fix": "date range",
            },
        ],
        optimized_query=(
            "SELECT * FROM orders WHERE customer_id = 5000 "
            "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01';"
        ),
        summary="Made the CAST and year() predicates sargable; flagged SELECT *.",
        estimated_improvement="~10x faster",
        approved=False,
    )


class TestReset:
    def test_reset_returns_task_observation(self, env):
        obs = env.reset(task_id="task_1_basic_antipatterns")
        assert isinstance(obs, SQLOptimObservation)
        assert obs.task_id == "task_1_basic_antipatterns"
        assert obs.step_count == 0
        assert obs.reward is None
        assert obs.done is False
        assert obs.sql_query  # a query is presented

    def test_reset_defaults_to_first_task(self, env):
        obs = env.reset()
        assert obs.task_id == DEFAULT_TASK_ID

    def test_reset_unknown_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="does_not_exist")


class TestStep:
    def test_step_rewards_a_faster_correct_rewrite(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        obs = env.step(_good_action())
        assert isinstance(obs.reward, float)
        assert obs.reward > 0.0
        assert "reward_breakdown" in obs.metadata
        assert obs.metadata["execution"]["speedup"] > 1.0  # real DuckDB speedup

    def test_step_accumulates_issue_context(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        obs = env.step(_good_action())
        assert set(obs.issues_found_so_far) >= {"select_star", "non_sargable_cast"}

    def test_episode_terminates_at_max_steps(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        max_steps = TASKS["task_1_basic_antipatterns"]["max_steps"]
        empty = SQLOptimAction(optimized_query="")  # no reward, forces max_steps
        done = False
        for _ in range(max_steps):
            done = env.step(empty).done
        assert done is True

    def test_step_after_done_raises(self, env):
        env.reset(task_id="task_1_basic_antipatterns")
        max_steps = TASKS["task_1_basic_antipatterns"]["max_steps"]
        empty = SQLOptimAction(optimized_query="")
        for _ in range(max_steps):
            env.step(empty)
        with pytest.raises(RuntimeError, match="Episode finished"):
            env.step(empty)


class TestState:
    def test_state_tracks_episode(self, env):
        env.reset(task_id="task_2_correlated_subqueries")
        state = env.state
        assert isinstance(state, SQLOptimState)
        assert state.step_count == 0
        assert state.episode_done is False
        assert state.cumulative_reward == 0.0


class TestModels:
    def test_action_roundtrips(self):
        action = _good_action()
        dumped = action.model_dump()
        assert dumped["optimized_query"].startswith("SELECT *")
        assert SQLOptimAction(**dumped) == action

    def test_observation_carries_base_fields(self):
        obs = SQLOptimObservation(task_id="t", reward=0.5, done=True)
        assert obs.reward == 0.5 and obs.done is True


class TestClientParsing:
    """The client's pure helpers, without opening a WebSocket."""

    def _client(self):
        return SQLOptimEnv(base_url="http://localhost:8000")

    def test_step_payload(self):
        payload = self._client()._step_payload(_good_action())
        assert payload["optimized_query"].startswith("SELECT *")
        assert payload["approved"] is False

    def test_parse_result(self):
        client = self._client()
        result = client._parse_result(
            {
                "observation": {"task_id": "t", "sql_query": "SELECT 1"},
                "reward": 0.42,
                "done": True,
                "metadata": {"feedback": "ok"},
            }
        )
        assert isinstance(result.observation, SQLOptimObservation)
        assert result.reward == 0.42
        assert result.done is True
        assert result.metadata == {"feedback": "ok"}

    def test_parse_state(self):
        state = self._client()._parse_state({"task_id": "t", "step_count": 3})
        assert isinstance(state, SQLOptimState)
        assert state.task_id == "t" and state.step_count == 3


class TestExecutionSafetyAndCorrectness:
    """Regression tests for the execution-engine review findings.

    Cover DB isolation, checksum robustness, single-compare-per-step, stale
    execution on empty rewrites, and the task-definition corrections.
    """

    def test_mutating_rewrite_is_rejected_and_db_intact(self, executor):
        """A DROP/DELETE rewrite is rejected and never mutates the shared DB."""
        before = executor.table_stats["orders"]
        for bad in (
            "DROP TABLE orders",
            "DELETE FROM orders",
            "SELECT 1; DROP TABLE orders",
        ):
            res = executor.compare("SELECT 1", bad)
            assert res["optimized_error"], f"not rejected: {bad}"
            assert res["results_match"] is False
        assert executor.table_stats["orders"] == before  # nothing persisted

    def test_read_only_rewrite_still_runs(self, executor):
        """A legitimate SELECT rewrite executes normally."""
        res = executor.compare(
            "SELECT COUNT(*) FROM orders", "SELECT COUNT(*) FROM orders"
        )
        assert res["optimized_error"] is None
        assert res["results_match"] is True

    def test_trailing_semicolon_large_result_checksum(self, executor):
        """A `;`-terminated query over >50k rows uses the checksum path correctly."""
        q = "SELECT * FROM events;"  # 1M rows -> checksum branch, trailing ';'
        res = executor.compare(q, q)
        assert res["results_match"] is True
        assert res["original_rows"] == 1_000_000

    def test_task3_equality_rewrite_is_result_preserving(self, executor):
        """Task 3's intended equality rewrite returns exactly the same rows.

        (Speedup is not asserted — wall-clock timing is too noisy for a stable
        threshold; result-preservation is the property that matters here.)
        """
        original = TASKS["task_3_wildcard_scan"]["sql_query"]
        rewrite = original.replace(
            "WHERE event_type LIKE '%purchase%'\n   OR event_type LIKE '%buy%';",
            "WHERE event_type = 'purchase';",
        )
        assert rewrite != original  # the replacement actually happened
        res = executor.compare(original, rewrite)
        assert res["results_match"] is True
        assert res["optimized_error"] is None

    def test_task5_is_deterministic(self, executor):
        """Task 5's windows are deterministic, so a result-preserving rewrite can match."""
        q = TASKS["task_5_window_functions"]["sql_query"]
        assert executor.compare(q, q)["results_match"] is True

    def test_task4_subqueries_labeled_uncorrelated(self):
        """Task 4 no longer mislabels its uncorrelated scalar subqueries as correlated."""
        types = [
            g["type"] for g in TASKS["task_4_implicit_join"]["ground_truth_issues"]
        ]
        assert "uncorrelated_scalar_subquery_avg" in types
        assert "uncorrelated_scalar_subquery_max" in types
        blob = str(TASKS["task_4_implicit_join"]).lower()
        assert "uncorrelated" in blob
        assert "per group" not in blob  # the "recomputed per group" mislabel is gone

    def test_step_runs_compare_once(self, monkeypatch):
        """`step` reuses the grader's comparison instead of running DuckDB twice."""
        from envs.sql_optim_env.server import executor as ex_mod

        ex = ex_mod.get_executor()
        real = ex.compare
        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(ex, "compare", counting)
        e = SQLOptimEnvironment()
        e.reset(task_id="task_1_basic_antipatterns")
        e.step(_good_action())
        assert calls["n"] == 1  # was 2 before the dedup fix

    def test_empty_rewrite_does_not_report_stale_execution(self):
        """An empty rewrite clears `execution` instead of keeping the prior step's."""
        e = SQLOptimEnvironment()
        e.reset(task_id="task_1_basic_antipatterns")
        obs1 = e.step(
            SQLOptimAction(
                suggestions=[{"issue_type": "select_star", "severity": "low"}],
                optimized_query="SELECT id FROM orders WHERE customer_id = 5000",
                summary="partial",
                approved=False,
            )
        )
        assert obs1.metadata["execution"] is not None
        assert obs1.done is False
        obs2 = e.step(
            SQLOptimAction(optimized_query="", summary="no rewrite", approved=False)
        )
        assert obs2.metadata["execution"] is None  # not the stale step-1 comparison

    def test_valid_rewrites_run_including_comments_and_literals(self, executor):
        """Valid read-only rewrites execute — no structural pre-check false-rejects.

        The read-only connection is the only gate, so rewrites using function
        names (`REPLACE`), string literals containing `set`/`;`, or a leading SQL
        comment all run instead of being rejected up front.
        """
        for good in (
            "SELECT REPLACE(status, 'a', 'b') FROM orders LIMIT 1",
            "SELECT COUNT(*) FROM orders WHERE status = 'set'",
            "SELECT COUNT(*) FROM orders WHERE status = 'a;b'",
            "-- optimized\nSELECT COUNT(*) FROM orders",
        ):
            res = executor.compare("SELECT COUNT(*) FROM orders", good)
            assert res["optimized_error"] is None, f"falsely rejected: {good!r}"

    def test_issue_detection_ignores_the_rewrite_sql(self):
        """Echoing the slow SQL earns no issue-detection credit without analysis.

        Anti-pattern tokens live in the query itself; scoring detection off the
        rewrite would let an agent farm the slice by copying the original query.
        """
        from envs.sql_optim_env.server.graders import grade

        original = TASKS["task_1_basic_antipatterns"]["sql_query"]
        # Rewrite echoes the original SQL, but the analysis fields are empty.
        action = SQLOptimAction(
            suggestions=[], optimized_query=original, summary="", approved=False
        )
        reward = grade(TASKS["task_1_basic_antipatterns"], action)
        assert reward.breakdown["issue_detection"] == 0.0

    def test_read_only_connection_rejects_writes_at_engine_level(self, executor):
        """Even a DML-in-CTE form is rejected — the connection is truly read-only."""
        before = executor.table_stats["orders"]
        res = executor.compare("SELECT 1", "WITH d AS (SELECT 1) DELETE FROM orders")
        assert res["optimized_error"]  # engine (or structural gate) rejects it
        assert executor.table_stats["orders"] == before

    def test_task1_result_preserving_rewrite_matches(self, executor):
        """Task 1's intended rewrite (sargable predicates, SELECT * kept) matches."""
        original = TASKS["task_1_basic_antipatterns"]["sql_query"]
        rewrite = (
            "SELECT * FROM orders WHERE customer_id = 5000 "
            "AND created_at >= DATE '2024-01-01' AND created_at < DATE '2025-01-01'"
        )
        res = executor.compare(original, rewrite)
        assert res["results_match"] is True

    def test_task2_last_order_is_deterministic(self, executor):
        """Task 2's `last_order_amount` subquery has a unique tie-breaker."""
        q = TASKS["task_2_correlated_subqueries"]["sql_query"]
        assert executor.compare(q, q)["results_match"] is True

    def test_speedup_requires_correctness(self):
        """A fast but wrong rewrite earns no speedup credit (only when results match)."""
        from envs.sql_optim_env.server.graders import grade

        task = TASKS["task_5_window_functions"]
        # Trivially fast, but wrong results.
        wrong = SQLOptimAction(optimized_query="SELECT 1", summary="x" * 60)
        r_wrong = grade(task, wrong)
        assert r_wrong.breakdown["execution_speedup"] == 0.0
        assert r_wrong.execution["results_match"] is False

    def test_concurrent_compare_is_safe(self, executor):
        """Concurrent access to the shared connection does not corrupt or crash."""
        import threading

        errors = []

        def worker():
            try:
                for _ in range(3):
                    executor.compare(
                        "SELECT COUNT(*) FROM orders",
                        "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
                    )
                    assert executor.table_stats["orders"] == 500_000
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
