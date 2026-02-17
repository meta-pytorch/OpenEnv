# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for OpenClawAdapter (RFC 005)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openenv.core.harnesses import HarnessConfig, HarnessEventType, HarnessTransport
from openenv.core.harnesses.adapters.openclaw import OpenClawAdapter


@pytest.fixture
def openclaw_config():
    return HarnessConfig(
        name="openclaw",
        command=["openclaw", "run"],
        transport=HarnessTransport.STDIO,
        model="claude-sonnet-4-20250514",
        api_key_env_var="ANTHROPIC_API_KEY",
        session_timeout_s=10.0,
    )


@pytest.fixture
def adapter(openclaw_config):
    return OpenClawAdapter(config=openclaw_config)


class TestOpenClawAdapterImport:
    """Test imports."""

    def test_import_from_adapters(self):
        from openenv.core.harnesses.adapters import OpenClawAdapter

        assert OpenClawAdapter is not None

    def test_inherits_from_harness_adapter(self):
        from openenv.core.harnesses import HarnessAdapter

        assert issubclass(OpenClawAdapter, HarnessAdapter)


class TestOpenClawAdapterInit:
    """Test initialization."""

    def test_stores_config(self, adapter, openclaw_config):
        assert adapter.config is openclaw_config

    def test_process_starts_none(self, adapter):
        assert adapter._process is None


class TestOpenClawAdapterInjectTools:
    """Test MCP tool injection via config file."""

    @pytest.mark.asyncio
    async def test_inject_creates_config_file(self, adapter, tmp_path):
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(tmp_path / ".openclaw" / "openclaw.json"),
            }
        )

        class FakeTool:
            name = "my_tool"

        await adapter.inject_tools([FakeTool()])

        config_path = tmp_path / ".openclaw" / "openclaw.json"
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert "mcpServers" in data
        assert "openenv" in data["mcpServers"]
        assert data["mcpServers"]["openenv"]["command"] == "openenv-mcp-bridge"

    @pytest.mark.asyncio
    async def test_inject_merges_with_existing_config(self, adapter, tmp_path):
        config_path = tmp_path / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {"existing": {"command": "existing-server"}},
                    "otherSetting": True,
                }
            )
        )

        adapter.config = adapter.config.model_copy(
            update={"mcp_config_path": str(config_path)}
        )

        class FakeTool:
            name = "env_tool"

        await adapter.inject_tools([FakeTool()])

        data = json.loads(config_path.read_text())
        # Should preserve existing entries
        assert "existing" in data["mcpServers"]
        assert "openenv" in data["mcpServers"]
        assert data["otherSetting"] is True

    @pytest.mark.asyncio
    async def test_inject_no_tools_skips_file(self, adapter, tmp_path):
        adapter.config = adapter.config.model_copy(
            update={
                "mcp_config_path": str(tmp_path / ".openclaw" / "openclaw.json"),
            }
        )
        await adapter.inject_tools([])
        config_path = tmp_path / ".openclaw" / "openclaw.json"
        assert not config_path.exists()


class TestOpenClawAdapterLifecycle:
    """Test start/stop with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_start_launches_process(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ) as mock_exec:
            await adapter.start("/workspace")

            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[0][0] == "openclaw"
            assert "--model" in call_args[0]
            assert "claude-sonnet-4-20250514" in call_args[0]

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, adapter):
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(return_value=0)

        adapter._process = mock_process
        await adapter.stop()

        mock_process.terminate.assert_called_once()
        assert adapter._process is None

    @pytest.mark.asyncio
    async def test_is_alive_false_when_no_process(self, adapter):
        assert await adapter.is_alive() is False

    @pytest.mark.asyncio
    async def test_is_alive_true_when_running(self, adapter):
        mock_process = MagicMock()
        mock_process.returncode = None
        adapter._process = mock_process
        assert await adapter.is_alive() is True

    @pytest.mark.asyncio
    async def test_is_alive_false_when_exited(self, adapter):
        mock_process = MagicMock()
        mock_process.returncode = 0
        adapter._process = mock_process
        assert await adapter.is_alive() is False


class TestOpenClawAdapterSendMessage:
    """Test message sending with mocked process I/O."""

    @pytest.mark.asyncio
    async def test_send_message_basic(self, adapter):
        response_data = json.dumps({"response": "Done.", "done": True}) + "\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=response_data.encode())

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("Fix the bug")

        assert resp.response == "Done."
        assert resp.done is True
        assert len(resp.events) >= 2  # LLM_REQUEST + TURN_COMPLETE

    @pytest.mark.asyncio
    async def test_send_message_with_tool_calls(self, adapter):
        response_data = (
            json.dumps(
                {
                    "response": "Fixed it.",
                    "done": True,
                    "tool_calls": [
                        {"tool_name": "read_file", "arguments": {"path": "auth.py"}},
                        {"tool_name": "write_file", "arguments": {"path": "auth.py"}},
                    ],
                }
            )
            + "\n"
        )

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=response_data.encode())

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("Fix the bug")

        tool_events = [e for e in resp.events if e.type == HarnessEventType.TOOL_CALL]
        assert len(tool_events) == 2
        assert tool_events[0].data["tool_name"] == "read_file"
        assert tool_events[1].data["tool_name"] == "write_file"

    @pytest.mark.asyncio
    async def test_send_message_plain_text_response(self, adapter):
        """Test handling of non-JSON plain text response."""
        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=b"Just a plain response\n")

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        resp = await adapter.send_message("Hello")

        assert resp.response == "Just a plain response"
        assert resp.done is False

    @pytest.mark.asyncio
    async def test_send_message_raises_when_not_running(self, adapter):
        with pytest.raises(RuntimeError, match="not running"):
            await adapter.send_message("test")


class TestOpenClawAdapterStreaming:
    """Test streaming interface."""

    @pytest.mark.asyncio
    async def test_streaming_yields_events(self, adapter):
        response_data = json.dumps({"response": "Done.", "done": True}) + "\n"

        mock_stdin = AsyncMock()
        mock_stdin.write = MagicMock()
        mock_stdin.drain = AsyncMock()

        mock_stdout = AsyncMock()
        mock_stdout.readline = AsyncMock(return_value=response_data.encode())

        mock_process = MagicMock()
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        mock_process.returncode = None
        adapter._process = mock_process

        events = []
        async for event in adapter.send_message_streaming("test"):
            events.append(event)

        assert len(events) >= 2
        assert events[-1].type == HarnessEventType.TURN_COMPLETE


class TestOpenClawEndToEnd:
    """Test OpenClaw adapter with HarnessEnvironment."""

    def test_works_with_harness_environment(self, adapter):
        """Verify OpenClawAdapter can be used with HarnessEnvironment."""
        from openenv.core.harnesses import HarnessEnvironment

        env = HarnessEnvironment(adapter=adapter)
        assert env.adapter is adapter
