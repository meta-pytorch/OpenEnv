# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The two upstream Harbor defects that cost `hermes` and `openclaw` their ATIF trajectory.

Both failed the same way: no error, no exception, no missing file -- just an agent that quietly
produced no trace, so every rollout reported `atif=none` and the cross-check silently did not exist.
Neither is detectable from a passing rollout, which is why they are pinned here.
"""

from __future__ import annotations

import asyncio
import json

import pytest

install_fixes = pytest.importorskip("openenv.harbor.install_fixes")
openclaw_mod = pytest.importorskip("harbor.agents.installed.openclaw")
hermes_mod = pytest.importorskip("harbor.agents.installed.hermes")

InterceptHermes = install_fixes.InterceptHermes
TRIM = install_fixes._OPENCLAW_TRIM_TRAILING_LOG
OpenClaw = openclaw_mod.OpenClaw
Hermes = hermes_mod.Hermes

_CONTAINER_PATH = "/logs/agent/openclaw.txt"
_SESSION_FILE = "/root/.openclaw/agents/main/sessions/790c93f1.jsonl"

# The shape openclaw actually produces: a pretty-printed envelope whose closing brace sits at
# column 0, and whose LAST nested object is the `completion` block. Both details matter below, and
# the key ORDER is taken from a real capture file (`payloads` first, `meta` last) because the
# backwards-scan trap depends on which nested object happens to be last.
_ENVELOPE = {
    "payloads": [],
    "meta": {
        "agentMeta": {"sessionId": "790c93f1", "sessionFile": _SESSION_FILE},
        "completion": {"stopReason": "stop", "finishReason": "stop"},
    },
}
# Harbor merges the agent's stderr into the same file with `2>&1`, so this lands after the JSON.
_TRAILING_LOG = (
    "[agents/agent-command] [agent] run 9e921697-bfe9-4266-ad60-6e9f65d0de5e "
    "ended with stopReason=stop"
)


def _capture_file(with_trailing_log: bool = True) -> str:
    body = json.dumps(_ENVELOPE, indent=2)
    return f"{body}\n{_TRAILING_LOG}\n" if with_trailing_log else f"{body}\n"


def _run_trim(tmp_path) -> str:
    """Execute the real production trim script against a temp file, not a copy of its logic."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    # If the constant is ever reworded, the substitution stops matching and this test would
    # silently exercise nothing. Fail instead.
    assert script != TRIM, f"{_CONTAINER_PATH!r} no longer appears in the trim script"
    target.write_text(_capture_file(), encoding="utf-8")
    exec(compile(script, "<trim>", "exec"), {})
    return target.read_text(encoding="utf-8")


# --- openclaw ---------------------------------------------------------------
def test_harbor_cannot_parse_its_own_capture_file_when_openclaw_logs_after_the_json():
    """The bug itself: one stderr line after the envelope and Harbor's parser gives up.

    `_load_json_object` requires the JSON object to consume the entire remaining suffix, but Harbor's
    own `2>&1` is what put a non-JSON line there. Returning None means `populate_context_post_run`
    returns at `if not envelope` and no `trajectory.json` is ever written.
    """
    assert OpenClaw._load_json_object(_capture_file()) is None


def test_trimming_the_trailing_log_line_makes_harbors_own_parser_succeed(tmp_path):
    """The fix, stated as the only thing it is allowed to be: Harbor's parser does the parsing.

    The subclass removes the trailing lines and nothing else, so the envelope that comes back is
    Harbor's own -- including `agentMeta.sessionFile`, which is what the session copy needs.
    """
    parsed = OpenClaw._load_json_object(_run_trim(tmp_path))

    assert parsed is not None
    assert parsed["meta"]["agentMeta"]["sessionFile"] == _SESSION_FILE


def test_trim_leaves_an_already_clean_capture_file_untouched(tmp_path):
    """A run whose stopReason is `end_turn` logs nothing, so the file is already parseable."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    clean = _capture_file(with_trailing_log=False)
    target.write_text(clean, encoding="utf-8")

    exec(compile(script, "<trim>", "exec"), {})

    assert target.read_text(encoding="utf-8") == clean


def test_trim_survives_a_capture_file_with_no_envelope_at_all(tmp_path):
    """An agent that died before emitting JSON must not turn into a crash in our override."""
    target = tmp_path / "openclaw.txt"
    script = TRIM.replace(_CONTAINER_PATH, str(target))
    garbage = "openclaw: command not found\n"
    target.write_text(garbage, encoding="utf-8")

    exec(compile(script, "<trim>", "exec"), {})

    assert target.read_text(encoding="utf-8") == garbage


def test_a_backwards_scan_without_the_suffix_rule_latches_onto_the_wrong_object():
    """Why the fix trims text instead of loosening the parser -- the obvious loosening is wrong.

    Dropping Harbor's "must consume the suffix" rule looks like the one-line fix. It is not: the scan
    walks backwards, so the first thing that decodes is the LAST nested object, and `completion`
    decodes perfectly. The caller then gets a dict with no `meta` at all and builds a degenerate
    2-step trajectory from it -- which still reports `atif=match`, because `reconcile` downgrades a
    trace carrying no token counts instead of failing it. A silently wrong trace is worse than none.
    """
    text = _capture_file().strip()
    decoder = json.JSONDecoder()
    found = None
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(obj, dict):
            found = obj
            break

    assert found == {"stopReason": "stop", "finishReason": "stop"}
    assert "meta" not in found


# --- hermes -----------------------------------------------------------------
# Driven through `asyncio.run` rather than written as `async def` tests: `asyncio_mode = "auto"` in
# pyproject only takes effect when pytest-asyncio is installed, and it is not in every environment
# that runs this suite. A test that silently does not execute would defeat the purpose.
def test_hermes_session_is_exported_without_the_source_filter(monkeypatch):
    """`--source cli` matches nothing, so the export must be re-run without any filter.

    In hermes-agent ANY filter routes export through `list_prune_candidates()`, whose WHERE clause
    starts `s.ended_at IS NOT NULL` -- and a `chat -q` session never sets `ended_at`. The flag
    therefore selects zero rows for exactly the sessions Harbor creates, and the command still exits
    0, so nothing anywhere reports a problem.
    """
    commands: list[str] = []

    async def fake_super_run(self, instruction, environment, context):
        return None

    async def fake_exec_as_agent(
        self, environment, command, env=None, timeout_sec=None
    ):
        commands.append(command)

    monkeypatch.setattr(Hermes, "run", fake_super_run)
    monkeypatch.setattr(
        InterceptHermes, "exec_as_agent", fake_exec_as_agent, raising=False
    )

    agent = object.__new__(InterceptHermes)
    asyncio.run(InterceptHermes.run(agent, "solve it", object(), object()))

    assert len(commands) == 1, commands
    assert "hermes sessions export /logs/agent/hermes-session.jsonl" in commands[0]
    assert "--source" not in commands[0]


def test_a_failed_hermes_export_does_not_fail_the_rollout(monkeypatch):
    """The trajectory is a cross-check, not the product. Losing it must not discard a graded run."""
    warnings: list[str] = []

    async def fake_super_run(self, instruction, environment, context):
        return None

    async def boom(self, environment, command, env=None, timeout_sec=None):
        raise RuntimeError("sandbox went away")

    class _Logger:
        def warning(self, message, *args):
            warnings.append(message % args if args else message)

    monkeypatch.setattr(Hermes, "run", fake_super_run)
    monkeypatch.setattr(InterceptHermes, "exec_as_agent", boom, raising=False)

    agent = object.__new__(InterceptHermes)
    agent.logger = _Logger()
    asyncio.run(InterceptHermes.run(agent, "solve it", object(), object()))

    assert warnings and "sandbox went away" in warnings[0]
