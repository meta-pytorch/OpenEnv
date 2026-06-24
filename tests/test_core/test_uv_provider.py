# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for UVProvider readiness handling.

These tests cover the health-poll loop and the ``context_timeout_s`` knob
without launching a real ``uv`` subprocess or hitting the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from openenv.core.containers.runtime import uv_provider
from openenv.core.containers.runtime.uv_provider import _poll_health, UVProvider


@pytest.fixture()
def provider():
    """A UVProvider that skips the real `uv --version` check."""
    with patch.object(uv_provider, "_check_uv_installed"):
        return UVProvider(project_path="/tmp/some-env", context_timeout_s=300.0)


class TestPollHealth:
    """`_poll_health` must back off between attempts, not busy-spin."""

    def test_sleeps_between_refused_connections(self):
        """A refused connection returns instantly; without a pause between
        attempts the loop would peg a CPU core for the whole timeout window.
        The loop must therefore sleep on the connection-error path too.
        """
        with (
            patch("requests.get", side_effect=requests.ConnectionError("refused")),
            patch.object(uv_provider.time, "sleep") as mock_sleep,
        ):
            with pytest.raises(TimeoutError, match="did not become ready"):
                _poll_health("http://localhost:9/health", timeout_s=0.05)

        assert mock_sleep.called, (
            "health poll busy-spun on connection errors without sleeping"
        )

    def test_returns_when_healthy(self):
        """Returns as soon as /health responds 200."""
        ok = MagicMock(status_code=200)
        with (
            patch("requests.get", return_value=ok) as mock_get,
            patch.object(uv_provider.time, "sleep"),
        ):
            _poll_health("http://localhost:9/health", timeout_s=5.0)

        mock_get.assert_called()


class TestWaitForReadyTimeout:
    """`wait_for_ready()` must honor the configured `context_timeout_s`."""

    def test_defaults_to_context_timeout(self, provider):
        """Called with no argument (as EnvClient does), the readiness wait uses
        the provider's `context_timeout_s` rather than a hardcoded default.
        """
        provider._base_url = "http://localhost:8000"
        provider._process = None

        with patch.object(uv_provider, "_poll_health") as mock_poll:
            provider.wait_for_ready()

        mock_poll.assert_called_once()
        assert mock_poll.call_args.kwargs["timeout_s"] == 300.0

    def test_explicit_timeout_overrides(self, provider):
        """An explicit timeout_s still wins over context_timeout_s."""
        provider._base_url = "http://localhost:8000"
        provider._process = None

        with patch.object(uv_provider, "_poll_health") as mock_poll:
            provider.wait_for_ready(timeout_s=5.0)

        assert mock_poll.call_args.kwargs["timeout_s"] == 5.0
