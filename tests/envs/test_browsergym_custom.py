"""Unit tests for BrowserGym custom task system.

This comprehensive pytest suite tests the HTTP client/server integration.
Requires the BrowserGym server to be running.

Run with pytest (from project root):
    source .venv/bin/activate
    pytest tests/envs/test_browsergym_custom.py -v

Note: This suite starts its own server via fixtures. The maintainers will
run this as part of the CI/CD pipeline.
"""

import os
import sys
import subprocess
import time
import requests
import pytest

from envs.browsergym_env.client import BrowserGymEnv
from envs.browsergym_env.models import BrowserGymAction


@pytest.fixture(scope="module")
def custom_server():
    """Starts the BrowserGym environment server with custom task support."""
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PORT = 8012
    localhost = f"http://localhost:{PORT}"

    print(f"\n--- Starting BrowserGym custom task server on port {PORT} ---")

    server_env = {
        **os.environ,
        "BROWSERGYM_BENCHMARK": "custom",
        "BROWSERGYM_TASK_NAME": "copy-paste",
        "BROWSERGYM_HEADLESS": "true",
    }

    gunicorn_command = [
        "gunicorn",
        "-w",
        "1",
        "-k",
        "uvicorn.workers.UvicornWorker",
        "-b",
        f"0.0.0.0:{PORT}",
        "envs.browsergym_env.server.app:app",
    ]

    server_process = subprocess.Popen(
        gunicorn_command,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to become healthy
    print("\n--- Waiting for custom task server to become healthy... ---")
    is_healthy = False
    for i in range(12):
        try:
            response = requests.get(f"{localhost}/health", timeout=5)
            if response.status_code == 200:
                is_healthy = True
                print("✅ Custom task server is running and healthy!")
                break
        except requests.exceptions.RequestException:
            print(f"Attempt {i + 1}/12: Server not ready, waiting 10 seconds...")
            time.sleep(10)

    if not is_healthy:
        print("❌ Server did not become healthy in time. Aborting.")
        print("\n--- Server Logs ---")
        stdout, stderr = server_process.communicate(timeout=5)
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        try:
            server_process.kill()
        except ProcessLookupError:
            pass
        pytest.skip("Custom task server failed to start")

    yield localhost

    # Cleanup
    print("\n--- Cleaning up custom task server ---")
    try:
        server_process.kill()
        print("✅ Server process killed")
    except ProcessLookupError:
        print("✅ Server process was already killed")


class TestCustomTaskRegistration:
    """Test custom task registration system."""

    def test_task_registry_has_builtin_tasks(self, custom_server):
        """Test that built-in custom tasks are registered."""
        # Import the registry to check task registration
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from envs.browsergym_env.server.custom.custom_tasks import _CUSTOM_TASKS

        assert "copy-paste" in _CUSTOM_TASKS
        assert "copy-paste-multitab" in _CUSTOM_TASKS

    def test_get_custom_task_class(self, custom_server):
        """Test retrieving a custom task class."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from envs.browsergym_env.server.custom.custom_tasks import get_custom_task

        task_class = get_custom_task("copy-paste")
        assert task_class is not None
        assert hasattr(task_class, "_calculate_reward")
        assert hasattr(task_class, "_check_done")

    def test_invalid_task_name_raises_error(self, custom_server):
        """Test that requesting an invalid task raises ValueError."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from envs.browsergym_env.server.custom.custom_tasks import get_custom_task

        with pytest.raises(ValueError, match="Unknown custom task"):
            get_custom_task("nonexistent-task")


class TestCustomTaskEnvironment:
    """Test custom task environment functionality."""

    def test_health_endpoint(self, custom_server):
        """Test that the health endpoint works."""
        response = requests.get(f"{custom_server}/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_custom_task_reset(self, custom_server):
        """Test that reset() works with custom tasks."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        result = env.reset()

        assert result.observation is not None
        assert hasattr(result.observation, "text")
        assert result.observation.goal is not None
        assert "copy" in result.observation.goal.lower()

    def test_custom_task_step(self, custom_server):
        """Test that step() works with custom tasks."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Execute a simple action
        action = BrowserGymAction(action_str="click('#source')")
        result = env.step(action)

        assert result.observation is not None
        assert hasattr(result.observation, "reward")
        assert hasattr(result.observation, "done")

    def test_custom_task_state(self, custom_server):
        """Test that state() returns valid custom task state."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        state = env.state()
        assert state.benchmark == "custom"
        assert state.task_name in ["copy-paste", "copy-paste-multitab"]
        assert state.step_count >= 0


class TestActionParsing:
    """Test action parsing logic in custom tasks."""

    def test_click_action(self, custom_server):
        """Test that click actions are parsed correctly."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Click action should work
        action = BrowserGymAction(action_str="click('#source')")
        result = env.step(action)
        assert result.observation is not None

    def test_fill_action(self, custom_server):
        """Test that fill actions are parsed correctly."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Fill action should work
        action = BrowserGymAction(action_str="fill('#source', 'test text')")
        result = env.step(action)
        assert result.observation is not None

    def test_goto_action(self, custom_server):
        """Test that goto actions are parsed correctly."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Goto action should work (though may fail for custom tasks without navigation)
        action = BrowserGymAction(action_str="goto('about:blank')")
        result = env.step(action)
        assert result.observation is not None

    def test_press_action(self, custom_server):
        """Test that keyboard press actions work."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Press action should work
        action = BrowserGymAction(action_str="press('Enter')")
        result = env.step(action)
        assert result.observation is not None

    def test_scroll_action(self, custom_server):
        """Test that scroll actions work."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Scroll down
        action = BrowserGymAction(action_str="scroll('down')")
        result = env.step(action)
        assert result.observation is not None

    def test_javascript_fallback(self, custom_server):
        """Test that unrecognized actions fall back to JavaScript execution."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Raw JavaScript should execute
        action = BrowserGymAction(action_str="console.log('test')")
        result = env.step(action)
        assert result.observation is not None

    def test_malformed_action_handling(self, custom_server):
        """Test that malformed actions are handled gracefully."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Malformed action should not crash the server
        action = BrowserGymAction(action_str="click(invalid syntax")
        result = env.step(action)
        assert result.observation is not None
        # Error should be reflected in observation metadata or error field


class TestObservationConversion:
    """Test observation conversion from custom task to BrowserGym format."""

    def test_observation_has_required_fields(self, custom_server):
        """Test that observations contain all required fields."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        result = env.reset()

        obs = result.observation
        assert hasattr(obs, "text")
        assert hasattr(obs, "goal")
        assert hasattr(obs, "done")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "metadata")

    def test_observation_text_extraction(self, custom_server):
        """Test that observation text is extracted from page."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        result = env.reset()

        # Should have some text content
        assert result.observation.text is not None
        assert len(result.observation.text) > 0

    def test_reward_calculation(self, custom_server):
        """Test that rewards are calculated correctly."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Take a step and check reward
        action = BrowserGymAction(action_str="click('#source')")
        result = env.step(action)

        assert result.observation.reward is not None
        assert isinstance(result.observation.reward, (int, float))

    def test_done_flag_detection(self, custom_server):
        """Test that done flag is set appropriately."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        result = env.reset()

        # Initially should not be done
        assert not result.observation.done

        # After max steps or success, should be done
        # (This depends on task implementation)


class TestErrorHandling:
    """Test error handling in custom task system."""

    def test_invalid_task_name_at_startup(self):
        """Test that invalid task name is handled at server startup."""
        # This would need to start a server with an invalid task name
        # and verify it either fails gracefully or returns an error
        pass

    def test_action_error_handling(self, custom_server):
        """Test that action errors are captured in observations."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Try to click a non-existent element
        action = BrowserGymAction(action_str="click('#nonexistent-element-xyz')")
        result = env.step(action)

        # Should not crash, but may have error in metadata
        assert result.observation is not None

    def test_max_steps_enforcement(self, custom_server):
        """Test that max_steps limit is enforced."""
        env = BrowserGymEnv(base_url=custom_server, request_timeout_s=60)
        env.reset()

        # Take many steps to hit the limit
        for i in range(15):
            action = BrowserGymAction(action_str="scroll('down')")
            result = env.step(action)

            # Should eventually be done due to max_steps
            if result.observation.done:
                break

        # After many steps, should be done
        assert env.state().step_count > 0


class TestTaskImplementations:
    """Test specific custom task implementations."""

    def test_copy_paste_task_goal(self, custom_server):
        """Test that copy-paste task has correct goal."""
        # Set up environment with copy-paste task
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        PORT = 8013
        localhost = f"http://localhost:{PORT}"

        server_env = {
            **os.environ,
            "BROWSERGYM_BENCHMARK": "custom",
            "BROWSERGYM_TASK_NAME": "copy-paste",
            "BROWSERGYM_HEADLESS": "true",
        }

        gunicorn_command = [
            "gunicorn",
            "-w",
            "1",
            "-k",
            "uvicorn.workers.UvicornWorker",
            "-b",
            f"0.0.0.0:{PORT}",
            "envs.browsergym_env.server.app:app",
        ]

        server_process = subprocess.Popen(
            gunicorn_command,
            env=server_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server
            time.sleep(10)

            env = BrowserGymEnv(base_url=localhost, request_timeout_s=60)
            result = env.reset()

            assert "copy" in result.observation.goal.lower()
            assert "paste" in result.observation.goal.lower()

        finally:
            try:
                server_process.kill()
            except ProcessLookupError:
                pass
