# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the REPL Environment."""

import pytest

# Skip entire module if smolagents is not installed
pytest.importorskip("smolagents", reason="smolagents is not installed")

from repl_env.models import REPLAction, REPLObservation, REPLState, CodeBlockResult
from repl_env.server.repl_environment import REPLEnvironment
from repl_env.server.python_executor import PythonExecutor


class TestPythonExecutor:
    """Tests for the PythonExecutor class."""

    def test_basic_execution(self):
        """Test basic code execution."""
        executor = PythonExecutor()
        result = executor.execute("x = 1 + 1")
        assert result["success"]
        assert executor.get_variable("x") == 2

    def test_stdout_capture(self):
        """Test stdout is captured correctly."""
        executor = PythonExecutor()
        result = executor.execute("print('hello world')")
        assert result["success"]
        assert "hello world" in result["stdout"]

    def test_stderr_capture(self):
        """Test stderr is captured correctly via exception handling.

        Note: smolagents.LocalPythonExecutor blocks direct sys.stderr access,
        so we test stderr capture through exception handling instead.
        """
        executor = PythonExecutor()
        # Exceptions are captured in stderr
        result = executor.execute("raise RuntimeError('error message')")
        assert not result["success"]
        assert "error message" in result["stderr"] or "error message" in result["exception"]

    def test_exception_handling(self):
        """Test exception handling."""
        executor = PythonExecutor()
        result = executor.execute("raise ValueError('test error')")
        assert not result["success"]
        assert result["exception"] is not None
        assert "ValueError" in result["exception"]
        assert "test error" in result["exception"]

    def test_persistent_namespace(self):
        """Test that namespace persists across executions."""
        executor = PythonExecutor()
        executor.execute("x = 10")
        executor.execute("y = x * 2")
        assert executor.get_variable("x") == 10
        assert executor.get_variable("y") == 20

    def test_context_loading(self):
        """Test context loading."""
        executor = PythonExecutor()
        executor.set_context("Hello World")
        assert executor.get_variable("context") == "Hello World"

    def test_list_variables(self):
        """Test listing variables."""
        executor = PythonExecutor()
        executor.execute("a = 1")
        executor.execute("b = 2")
        variables = executor.list_variables()
        assert "a" in variables
        assert "b" in variables

    def test_output_truncation(self):
        """Test output truncation."""
        executor = PythonExecutor(max_output_length=100)
        result = executor.execute("print('x' * 500)")
        assert len(result["stdout"]) < 200  # Should be truncated

    def test_inject_function(self):
        """Test function injection."""
        executor = PythonExecutor()

        def custom_func(x):
            return x * 2

        executor.inject_function("double", custom_func)
        result = executor.execute("result = double(5)")
        assert result["success"]
        assert executor.get_variable("result") == 10

    def test_reset(self):
        """Test namespace reset."""
        executor = PythonExecutor()
        executor.execute("x = 10")
        executor.reset()
        assert executor.get_variable("x") is None


class TestREPLEnvironment:
    """Tests for the REPLEnvironment class."""

    def test_reset_without_context(self):
        """Test reset without context."""
        env = REPLEnvironment()
        obs = env.reset()
        assert not obs.done
        assert obs.iteration == 0
        assert obs.context_length == 0
        assert "answer" in obs.available_variables

    def test_reset_with_context(self):
        """Test reset with context."""
        env = REPLEnvironment(context="Hello World")
        obs = env.reset()
        assert not obs.done
        assert obs.context_length == 11
        assert "context" in obs.available_variables
        assert obs.context_preview == "Hello World"

    def test_reset_with_task_prompt(self):
        """Test reset with task prompt."""
        env = REPLEnvironment(task_prompt="Count the chars")
        obs = env.reset()
        assert obs.metadata["task_prompt"] == "Count the chars"

    def test_step_basic(self):
        """Test basic step execution."""
        env = REPLEnvironment(context="test context")
        env.reset()
        obs = env.step(REPLAction(code="result = len(context)"))
        assert obs.result.success
        assert "result" in obs.available_variables
        assert obs.iteration == 1

    def test_step_with_error(self):
        """Test step with code error."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="raise ValueError('test')"))
        assert not obs.result.success
        assert obs.result.exception is not None
        assert not obs.done

    def test_final_pattern_basic(self):
        """Test FINAL() pattern."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="print('FINAL(42)')"))
        assert obs.done
        assert obs.metadata["final_answer"] == "42"

    def test_final_var_pattern(self):
        """Test FINAL_VAR() pattern."""
        env = REPLEnvironment()
        env.reset()
        env.step(REPLAction(code="my_answer = 'the answer is 42'"))
        obs = env.step(REPLAction(code="print('FINAL_VAR(my_answer)')"))
        assert obs.done
        assert obs.metadata["final_answer"] == "the answer is 42"

    def test_answer_dict_pattern(self):
        """Test Prime Intellect style answer dict."""
        env = REPLEnvironment()
        env.reset()
        env.step(REPLAction(code="answer['content'] = 'my answer'"))
        obs = env.step(REPLAction(code="answer['ready'] = True"))
        assert obs.done
        assert obs.metadata["final_answer"] == "my answer"

    def test_explicit_final(self):
        """Test explicit is_final=True."""
        env = REPLEnvironment()
        env.reset()
        obs = env.step(REPLAction(code="", is_final=True, final_answer="explicit answer"))
        assert obs.done
        assert env.state.final_answer == "explicit answer"

    def test_max_iterations(self):
        """Test max iterations limit."""
        env = REPLEnvironment(max_iterations=2)
        env.reset()
        env.step(REPLAction(code="x = 1"))
        obs = env.step(REPLAction(code="x = 2"))
        assert obs.done
        assert "Maximum iterations" in obs.result.stdout

    def test_state_property(self):
        """Test state property."""
        env = REPLEnvironment(context="test")
        env.reset()
        state = env.state
        assert state.context == "test"
        assert state.iteration == 0

    def test_state_not_initialized(self):
        """Test state raises error when not initialized."""
        env = REPLEnvironment()
        with pytest.raises(RuntimeError):
            _ = env.state

    def test_reward_on_success(self):
        """Test reward on successful completion."""
        env = REPLEnvironment(reward_on_success=1.0)
        env.reset()
        obs = env.step(REPLAction(code="print('FINAL(done)')"))
        assert obs.reward == 1.0

    def test_reward_on_error(self):
        """Test reward on code error."""
        env = REPLEnvironment(reward_on_error=-0.1)
        env.reset()
        obs = env.step(REPLAction(code="raise ValueError()"))
        assert obs.reward == -0.1

    def test_close(self):
        """Test close cleans up resources."""
        env = REPLEnvironment()
        env.reset()
        env.close()
        assert env._state is None
        assert env._executor is None

    def test_get_metadata(self):
        """Test get_metadata returns correct info."""
        env = REPLEnvironment(max_iterations=50)
        metadata = env.get_metadata()
        assert metadata.name == "repl_env"
        assert metadata.version == "0.1.0"

    def test_llm_functions_injected(self):
        """Test LLM functions are injected when provided."""
        def mock_query(prompt):
            return f"Response to: {prompt}"

        def mock_batch(prompts):
            return [f"Response to: {p}" for p in prompts]

        env = REPLEnvironment(llm_query_fn=mock_query, llm_batch_fn=mock_batch)
        env.reset()

        # Test llm_query
        obs = env.step(REPLAction(code="result = llm_query('Hello')"))
        assert obs.result.success
        obs = env.step(REPLAction(code="print(result)"))
        assert "Response to: Hello" in obs.result.stdout

        # Test llm_batch
        obs = env.step(REPLAction(code="results = llm_batch(['A', 'B'])"))
        assert obs.result.success


class TestModels:
    """Tests for the data models."""

    def test_repl_action_defaults(self):
        """Test REPLAction default values."""
        action = REPLAction(code="x = 1")
        assert action.code == "x = 1"
        assert action.is_final is False
        assert action.final_answer is None

    def test_repl_action_final(self):
        """Test REPLAction with final flag."""
        action = REPLAction(code="", is_final=True, final_answer="42")
        assert action.is_final is True
        assert action.final_answer == "42"

    def test_code_block_result(self):
        """Test CodeBlockResult model."""
        result = CodeBlockResult(
            stdout="output",
            stderr="",
            locals_snapshot={"x": "1"},
            execution_time=0.01,
            success=True,
        )
        assert result.stdout == "output"
        assert result.success is True

    def test_repl_observation(self):
        """Test REPLObservation model."""
        obs = REPLObservation(
            result=CodeBlockResult(
                stdout="test",
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=True,
            ),
            context_length=100,
            available_variables=["x", "y"],
            iteration=5,
            max_iterations=30,
            done=False,
            reward=0.0,
        )
        assert obs.context_length == 100
        assert len(obs.available_variables) == 2
        assert obs.iteration == 5

    def test_repl_state(self):
        """Test REPLState model."""
        state = REPLState(
            episode_id="test-123",
            step_count=5,
            context="hello",
            task_prompt="count",
            iteration=3,
            max_iterations=30,
        )
        assert state.episode_id == "test-123"
        assert state.context == "hello"


class TestREPLEnvUnifiedClient:
    """Tests for the unified REPLEnv client (local mode)."""

    def test_local_mode_basic(self):
        """Test basic local mode execution."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            result = env.reset()
            assert not result.done
            assert result.observation.iteration == 0

            result = env.execute("x = 42")
            assert result.observation.result.success

            result = env.execute("print(f'FINAL({x})')")
            assert result.done
            assert env.state().final_answer == "42"

    def test_local_mode_with_context(self):
        """Test local mode with context."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            result = env.reset(context="Hello World", task_prompt="Count chars")
            assert result.observation.context_length == 11
            assert "context" in result.observation.available_variables

            result = env.execute("count = len(context)")
            assert result.observation.result.success

    def test_local_mode_with_llm_functions(self):
        """Test local mode with LLM functions."""
        from repl_env import REPLEnv

        def mock_query(prompt):
            return f"Response: {prompt[:20]}"

        def mock_batch(prompts):
            return [mock_query(p) for p in prompts]

        with REPLEnv(llm_query_fn=mock_query, llm_batch_fn=mock_batch) as env:
            result = env.reset(context="Test")
            assert "llm_query" in result.observation.available_variables
            assert "llm_batch" in result.observation.available_variables

            result = env.execute("r = llm_query('Hello')")
            assert result.observation.result.success

            result = env.execute("print(r)")
            assert "Response: Hello" in result.observation.result.stdout

    def test_local_mode_max_iterations(self):
        """Test local mode max iterations."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            result = env.reset(max_iterations=2)

            result = env.execute("x = 1")
            assert not result.done

            result = env.execute("x = 2")
            assert result.done
            assert "Maximum iterations" in result.observation.result.stdout

    def test_local_mode_rewards(self):
        """Test local mode reward configuration."""
        from repl_env import REPLEnv

        with REPLEnv(reward_on_success=1.0, reward_on_error=-0.5) as env:
            env.reset()

            # Error should give negative reward
            result = env.execute("raise ValueError()")
            assert result.reward == -0.5

            # Success should give positive reward
            result = env.execute("print('FINAL(done)')")
            assert result.reward == 1.0

    def test_execute_convenience_method(self):
        """Test execute() convenience method."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            env.reset()
            result = env.execute("x = 1 + 1")
            assert result.observation.result.success

    def test_submit_final_answer(self):
        """Test submit_final_answer() method."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            env.reset()
            result = env.submit_final_answer("my answer")
            assert result.done
            assert env.state().final_answer == "my answer"

    def test_state_method(self):
        """Test state() method."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            env.reset(context="test", task_prompt="do something")
            state = env.state()
            assert state.context == "test"
            assert state.task_prompt == "do something"

    def test_list_variables(self):
        """Test list_variables() method."""
        from repl_env import REPLEnv

        with REPLEnv() as env:
            env.reset()
            env.execute("my_var = 123")
            variables = env.list_variables()
            assert "my_var" in variables

    def test_context_manager(self):
        """Test context manager properly closes."""
        from repl_env import REPLEnv

        env = REPLEnv()
        with env:
            env.reset()
            env.execute("x = 1")
        # After exiting context, internal state should be cleaned up
        assert env._local_env is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
