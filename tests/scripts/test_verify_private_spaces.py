"""Tests for the Hugging Face Space verification helper."""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock, patch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import verify_private_spaces


def make_response(
    *,
    status_code: int = 200,
    content_type: str = "text/html; charset=utf-8",
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    return response


def test_gradio_web_ok_html_accepts_gradio_markers() -> None:
    response = make_response()

    assert verify_private_spaces.gradio_web_ok_html(
        response,
        "<html><body><gradio-app></gradio-app></body></html>",
    )


def test_gradio_web_ok_html_rejects_non_gradio_html() -> None:
    response = make_response()

    assert not verify_private_spaces.gradio_web_ok_html(
        response,
        "<html><title>404 - Hugging Face</title><body>Not Found</body></html>",
    )


def test_gradio_web_ok_reset_requires_observation_payload() -> None:
    response = make_response(content_type="application/json")

    assert verify_private_spaces.gradio_web_ok_reset(response, {"observation": {"text": "ok"}})
    assert not verify_private_spaces.gradio_web_ok_reset(response, {"state": {}})


@patch("verify_private_spaces.run_probe_request")
def test_probe_gradio_web_space_checks_root_and_reset(mock_run_probe_request) -> None:
    mock_run_probe_request.side_effect = (
        lambda session, base_url, headers, method, path, timeout, payload=None, ok_fn=None: {
            "method": method,
            "path": path,
            "payload": payload,
            "ok": True,
        }
    )

    results = verify_private_spaces.probe_gradio_web_space(
        Mock(),
        "https://example.com",
        {},
        5.0,
    )

    assert [result["path"] for result in results] == [
        "/",
        "/web",
        "/web/",
        "/health",
        "/metadata",
        "/schema",
        "/reset",
    ]
    assert results[-1]["method"] == "POST"
    assert results[-1]["payload"] is None


@patch("verify_private_spaces.probe_gradio_web_space")
def test_probe_space_dispatches_gradio_web(mock_probe_gradio_web_space) -> None:
    mock_probe_gradio_web_space.return_value = [{"ok": True}]

    result = verify_private_spaces.probe_space(
        "https://example.com",
        headers={},
        timeout=5.0,
        probe_profile="gradio_web",
    )

    assert result == [{"ok": True}]
    mock_probe_gradio_web_space.assert_called_once()
