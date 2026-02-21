# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for OpenClaw Environment."""

import pytest
import tempfile
from pathlib import Path

# Skip if dependencies aren't available
pytest.importorskip("fastmcp")

from envs.openclaw_env.server.openclaw_environment import OpenClawEnvironment
from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction


class TestOpenClawEnvironment:
    """Test suite for OpenClawEnvironment."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = OpenClawEnvironment(workspace_dir=tmpdir)
            environment.reset()
            yield environment

    def test_reset(self, env):
        """Test environment reset creates workspace."""
        obs = env.reset()
        
        assert obs.done is False
        assert obs.metadata["status"] == "ready"
        assert "workspace" in obs.metadata
        assert Path(obs.metadata["workspace"]).exists()

    def test_list_tools(self, env):
        """Test that all expected tools are available."""
        obs = env.step(ListToolsAction())
        
        tool_names = [t.name for t in obs.tools]
        
        # Check all expected tools are present
        expected_tools = [
            "read", "write", "edit", "exec",
            "web_search", "web_fetch",
            "memory_search", "memory_get"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    def test_exec_echo(self, env):
        """Test executing a simple echo command."""
        action = CallToolAction(
            tool_name="exec",
            arguments={"command": "echo 'hello world'"}
        )
        obs = env.step(action)
        
        assert obs.result is not None
        # The result contains structured_content with our dict
        result_data = obs.result.structured_content.get("result", obs.result.data)
        assert result_data["exit_code"] == 0
        assert "hello world" in result_data["stdout"]

    def test_write_and_read(self, env):
        """Test writing and reading a file."""
        # Write a file
        write_action = CallToolAction(
            tool_name="write",
            arguments={
                "path": "test.txt",
                "content": "Hello, OpenClaw!"
            }
        )
        write_obs = env.step(write_action)
        
        write_result = write_obs.result.structured_content.get("result", write_obs.result.data)
        assert write_result["success"] is True
        
        # Read the file
        read_action = CallToolAction(
            tool_name="read",
            arguments={"path": "test.txt"}
        )
        read_obs = env.step(read_action)
        
        read_result = read_obs.result.structured_content.get("result", read_obs.result.data)
        assert read_result["content"] == "Hello, OpenClaw!"

    def test_edit_file(self, env):
        """Test editing a file."""
        # Write initial content
        env.step(CallToolAction(
            tool_name="write",
            arguments={
                "path": "config.txt",
                "content": "DEBUG = False\nVERBOSE = True"
            }
        ))
        
        # Edit the file
        edit_action = CallToolAction(
            tool_name="edit",
            arguments={
                "path": "config.txt",
                "old_string": "DEBUG = False",
                "new_string": "DEBUG = True"
            }
        )
        edit_obs = env.step(edit_action)
        
        edit_result = edit_obs.result.structured_content.get("result", edit_obs.result.data)
        assert edit_result["success"] is True
        assert edit_result["replacements"] == 1
        
        # Verify the change
        read_obs = env.step(CallToolAction(
            tool_name="read",
            arguments={"path": "config.txt"}
        ))
        read_result = read_obs.result.structured_content.get("result", read_obs.result.data)
        assert "DEBUG = True" in read_result["content"]

    def test_read_nonexistent_file(self, env):
        """Test reading a file that doesn't exist."""
        action = CallToolAction(
            tool_name="read",
            arguments={"path": "nonexistent.txt"}
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert "error" in result

    def test_exec_with_workdir(self, env):
        """Test executing command in specific directory."""
        # Create a subdirectory with a file
        env.step(CallToolAction(
            tool_name="exec",
            arguments={"command": "mkdir -p subdir && echo 'test' > subdir/file.txt"}
        ))
        
        # Execute in that directory
        action = CallToolAction(
            tool_name="exec",
            arguments={
                "command": "cat file.txt",
                "workdir": "subdir"
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["exit_code"] == 0
        assert "test" in result["stdout"]

    def test_web_search_sandbox(self, env):
        """Test web search returns simulated results in sandbox mode."""
        action = CallToolAction(
            tool_name="web_search",
            arguments={
                "query": "python reinforcement learning",
                "count": 3
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert "results" in result
        assert len(result["results"]) == 3
        assert "note" in result  # Sandbox mode indicator

    def test_web_fetch_sandbox(self, env):
        """Test web fetch returns simulated content in sandbox mode."""
        action = CallToolAction(
            tool_name="web_fetch",
            arguments={
                "url": "https://example.com",
                "extract_mode": "markdown"
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert "content" in result
        assert result["url"] == "https://example.com"
        assert "note" in result  # Sandbox mode indicator

    def test_memory_search_empty(self, env):
        """Test memory search with no memory files."""
        action = CallToolAction(
            tool_name="memory_search",
            arguments={"query": "test query"}
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert "results" in result
        assert result["total_found"] == 0

    def test_memory_search_with_content(self, env):
        """Test memory search finds content in memory files."""
        # Create a memory file
        env.step(CallToolAction(
            tool_name="write",
            arguments={
                "path": "MEMORY.md",
                "content": "# Important Notes\n\nRemember to check the API endpoints.\n"
            }
        ))
        
        # Search for content
        action = CallToolAction(
            tool_name="memory_search",
            arguments={"query": "API endpoints"}
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["total_found"] >= 1
        assert "API endpoints" in result["results"][0]["snippet"]

    def test_step_count_increments(self, env):
        """Test that step count increments properly."""
        initial_count = env.state.step_count
        
        env.step(ListToolsAction())
        assert env.state.step_count == initial_count + 1
        
        env.step(CallToolAction(tool_name="exec", arguments={"command": "echo hi"}))
        assert env.state.step_count == initial_count + 2

    def test_episode_isolation(self, env):
        """Test that reset creates a fresh episode."""
        # Create a file in first episode
        env.step(CallToolAction(
            tool_name="write",
            arguments={"path": "episode1.txt", "content": "first episode"}
        ))
        
        # Reset to new episode
        obs = env.reset()
        new_workspace = obs.metadata["workspace"]
        
        # File from first episode should not exist
        read_obs = env.step(CallToolAction(
            tool_name="read",
            arguments={"path": "episode1.txt"}
        ))
        result = read_obs.result.structured_content.get("result", read_obs.result.data)
        assert "error" in result


class TestOpenClawEnvironmentEdgeCases:
    """Edge case tests for OpenClawEnvironment."""

    @pytest.fixture
    def env(self):
        """Create environment for edge case tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            environment = OpenClawEnvironment(workspace_dir=tmpdir)
            environment.reset()
            yield environment

    def test_exec_timeout(self, env):
        """Test command timeout handling."""
        action = CallToolAction(
            tool_name="exec",
            arguments={
                "command": "sleep 10",
                "timeout": 1
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_edit_string_not_found(self, env):
        """Test editing when old string doesn't exist."""
        env.step(CallToolAction(
            tool_name="write",
            arguments={"path": "test.txt", "content": "hello world"}
        ))
        
        action = CallToolAction(
            tool_name="edit",
            arguments={
                "path": "test.txt",
                "old_string": "not found",
                "new_string": "replacement"
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["success"] is False

    def test_write_creates_directories(self, env):
        """Test that write creates parent directories."""
        action = CallToolAction(
            tool_name="write",
            arguments={
                "path": "deep/nested/dir/file.txt",
                "content": "nested content"
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["success"] is True
        
        # Verify file exists
        read_obs = env.step(CallToolAction(
            tool_name="read",
            arguments={"path": "deep/nested/dir/file.txt"}
        ))
        read_result = read_obs.result.structured_content.get("result", read_obs.result.data)
        assert read_result["content"] == "nested content"

    def test_read_with_offset_and_limit(self, env):
        """Test reading specific line ranges."""
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        env.step(CallToolAction(
            tool_name="write",
            arguments={"path": "lines.txt", "content": content}
        ))
        
        action = CallToolAction(
            tool_name="read",
            arguments={
                "path": "lines.txt",
                "offset": 50,
                "limit": 10
            }
        )
        obs = env.step(action)
        
        result = obs.result.structured_content.get("result", obs.result.data)
        assert result["lines_read"] == 10
        assert "Line 50" in result["content"]
        assert "Line 59" in result["content"]
