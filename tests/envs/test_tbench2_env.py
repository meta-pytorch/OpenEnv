import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

# Add the project root to the path for envs imports, and envs/ itself so the
# server module's Spaces-style absolute imports (``tbench2_env.models``)
# resolve when this file runs standalone (in the full suite another test
# happens to insert envs/ first — don't rely on collection order).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import camel  # noqa: F401
except Exception:
    camel = None

from envs.tbench2_env.models import Tbench2Action
from envs.tbench2_env.server import tbench2_env_environment
from envs.tbench2_env.server.tbench2_env_environment import (
    Tbench2DockerEnvironment,
    Tbench2Environment,
)


class _FakeTerminalToolkit:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def shell_exec(self, **kwargs):
        return ""


def test_tbench2_reset_uses_default_task_id(monkeypatch, tmp_path: Path):
    """HTTP resets without kwargs should land on the default demo task."""
    task_dir = tmp_path / "headless-terminal"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Solve the terminal task.\n")

    monkeypatch.setattr(
        tbench2_env_environment,
        "_require_terminal_toolkit",
        lambda: _FakeTerminalToolkit,
    )

    env = Tbench2Environment(
        tasks_dir=str(tmp_path),
        output_dir=str(tmp_path / "runs"),
        default_task_id="headless-terminal",
    )

    observation = env.reset()

    assert observation.success is True
    assert observation.task_id == "headless-terminal"
    assert "terminal task" in observation.instruction


# ---------------------------------------------------------------------------
# Verifier-asset withholding (TB2_WITHHOLD_TESTS) and stage-at-verify
# ---------------------------------------------------------------------------


def _make_task_dir(root: Path, name: str = "my-task") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "tests" / "test_outputs.py").write_text("EXPECTED = 'secret42'\n")
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("echo the-answer\n")
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    (task / "instruction.md").write_text("do the thing\n")
    return task


@pytest.fixture()
def staged_paths(monkeypatch, tmp_path: Path):
    """Redirect the official fixed paths (/tests, /logs/verifier) into tmp."""
    tests = tmp_path / "stage" / "tests"
    logs = tmp_path / "stage" / "logs" / "verifier"
    monkeypatch.setattr(tbench2_env_environment, "_VERIFY_TESTS_DIR", str(tests))
    monkeypatch.setattr(tbench2_env_environment, "_VERIFIER_LOG_DIR", str(logs))
    return tests, logs


@pytest.fixture(autouse=True)
def _fresh_withhold_cache(monkeypatch):
    monkeypatch.setattr(Tbench2Environment, "_WITHHELD_TESTS", {})


def test_withhold_removes_assets_and_caches_tests(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)

    env._withhold_verifier_assets(task)

    assert not (task / "solution").exists()
    assert not (task / "tests").exists()
    assert (task / "task.toml").is_file()
    assert (task / "instruction.md").is_file()
    assert str(task) in Tbench2Environment._WITHHELD_TESTS

    # Second reset of the same task (dir already gone) must keep the cache.
    env._withhold_verifier_assets(task)
    assert str(task) in Tbench2Environment._WITHHELD_TESTS


def test_stage_from_memory_and_wipe_planted_files(tmp_path: Path, staged_paths):
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    # An agent pre-plants a conftest.py (pytest would auto-load it) and a
    # fake reward before verification.
    stage_tests.mkdir(parents=True)
    (stage_tests / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_logs.mkdir(parents=True)
    (stage_logs / "reward.txt").write_text("1\n")

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test.sh").is_file()
    assert (stage_tests / "test_outputs.py").read_text() == "EXPECTED = 'secret42'\n"
    assert not (stage_tests / "conftest.py").exists()
    assert not (stage_logs / "reward.txt").exists()
    assert stage_logs.is_dir() and not list(stage_logs.iterdir())


def test_stage_from_disk_without_gate(tmp_path: Path, staged_paths):
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert env._stage_tests_for_verify()

    assert (stage_tests / "test_outputs.py").is_file()
    # Without the gate the source checkout is never touched.
    assert (task / "tests").is_dir()
    assert (task / "solution").is_dir()


def test_stage_reports_missing_tests(tmp_path: Path, staged_paths):
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task

    assert not env._stage_tests_for_verify()


class _RecordingToolkit:
    def __init__(self, output: str):
        self.calls: list[dict] = []
        self.output = output

    def shell_exec(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_evaluate_canonical_from_withheld_copy(tmp_path: Path, staged_paths):
    """Full evaluate flow: staging from memory + canonical test.sh scoring."""
    stage_tests, stage_logs = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._workdir = str(tmp_path)
    env._terminal_toolkit = _RecordingToolkit(output="__TB2_REWARD__:1")
    env._withhold_verifier_assets(task)

    output, reward, info = env._evaluate_task()

    assert reward == 1.0
    assert info["harness"] == "tests/test.sh"
    # The verify command ran test.sh from the staged fixed path, and wrote its
    # log under the wiped verifier dir — not a fixed /tmp path that outlives it.
    command = env._terminal_toolkit.calls[0]["command"]
    assert f"{stage_tests}/test.sh" in command
    assert "/tmp/tb2_testsh.log" not in command
    assert f"{stage_logs}/testsh.log" in command
    # Both the staged tests and the verifier log/reward dir are gone once
    # scoring returns.
    assert not stage_tests.exists()
    assert not stage_logs.exists()


def test_evaluate_without_tests_scores_zero(tmp_path: Path, staged_paths):
    task = tmp_path / "empty-task"
    task.mkdir()
    env = Tbench2Environment(withhold_tests=False)
    env._task_dir = task
    env._terminal_toolkit = _RecordingToolkit(output="")

    output, reward, info = env._evaluate_task()

    assert reward == 0.0
    assert info["tests_passed"] is False
    assert env._terminal_toolkit.calls == []  # nothing was run


def test_stage_wipes_symlinked_fixed_path(tmp_path: Path, staged_paths):
    """A symlinked /tests must not let the official suite merge into a dir the
    agent controls (rmtree silently no-ops on a symlink; _hard_remove doesn't)."""
    stage_tests, _ = staged_paths
    task = _make_task_dir(tmp_path)
    env = Tbench2Environment(withhold_tests=True)
    env._task_dir = task
    env._withhold_verifier_assets(task)

    evil = tmp_path / "agent_controlled"
    evil.mkdir()
    (evil / "conftest.py").write_text("import sys; sys.exit(0)\n")
    stage_tests.parent.mkdir(parents=True, exist_ok=True)
    stage_tests.symlink_to(evil, target_is_directory=True)

    assert env._stage_tests_for_verify()

    assert not stage_tests.is_symlink()
    assert (stage_tests / "test.sh").is_file()
    # The official suite never leaked into the agent's directory.
    assert not (evil / "test.sh").exists()
    assert (evil / "conftest.py").exists()  # their dir left alone


def test_withhold_removes_symlinked_tests(tmp_path: Path):
    """tests/ as a symlink is removed too — the link, not just its target."""
    task = tmp_path / "task"
    task.mkdir()
    real = tmp_path / "real_tests"
    real.mkdir()
    (real / "test_outputs.py").write_text("x = 1\n")
    (task / "tests").symlink_to(real, target_is_directory=True)

    env = Tbench2Environment(withhold_tests=True)
    env._withhold_verifier_assets(task)

    assert not (task / "tests").exists()  # symlink gone
    assert real.exists()  # its target is outside the task dir; untouched
    # The cache captured the target's real files, not a bare link member that
    # would restore an empty suite (tar.add doesn't follow a top-level link).
    import io
    import tarfile

    blob = Tbench2Environment._WITHHELD_TESTS[str(task)]
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        assert any(n.endswith("test_outputs.py") for n in tar.getnames())


# ---------------------------------------------------------------------------
# Docker mode: copy exclusions + stage-at-verify (mock container, no docker)
# ---------------------------------------------------------------------------


class _FakeContainer:
    """Records exec/put_archive calls in one ordered event log."""

    def __init__(self, exec_output: bytes = b"__TB2_EXIT_CODE__:0\n"):
        self.events: list[tuple] = []
        self.exec_output = exec_output

    def put_archive(self, dest, data):
        self.events.append(("put", dest, data))
        return True

    def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
        self.events.append(("exec", cmd))
        return 0, self.exec_output


def _tar_names(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return tar.getnames()


def test_docker_copy_excludes_verifier_assets(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM debian:12\n")

    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._copy_dir_to_container(task, "/task", exclude_top=("solution", "tests"))

    ((kind, dest, data),) = container.events
    assert (kind, dest) == ("put", "/task")
    names = _tar_names(data)
    assert "task.toml" in names
    assert "environment/Dockerfile" in names
    assert not any(n.startswith(("solution", "tests")) for n in names)
    assert len(names) == len(set(names))  # recursive=False: no duplicates


def test_evaluate_docker_stages_tests_at_verify(tmp_path: Path):
    task = _make_task_dir(tmp_path)
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 1.0
    kinds = [e[0] for e in container.events]
    # rm -rf the agent-writable staging dir BEFORE the fresh copy goes in,
    # the copy lands before pytest runs, and tests are removed again after.
    assert kinds == ["exec", "put", "exec", "exec"]
    assert "rm -rf /task/tests" in container.events[0][1]
    assert container.events[1][1] == "/task/tests"
    assert "test.sh" in _tar_names(container.events[1][2])
    assert "pytest" in container.events[2][1]
    assert "rm -rf /task/tests" in container.events[3][1]


def test_evaluate_docker_cleans_up_when_scoring_raises(tmp_path: Path):
    """The staged /task/tests must not outlive a verify that errors out —
    step() reports the failure without ending the episode, so the agent
    keeps its session."""
    task = _make_task_dir(tmp_path)

    class _ExplodingContainer(_FakeContainer):
        def exec_run(self, cmd, workdir=None, stdout=True, stderr=True):
            if "pytest" in cmd:
                self.events.append(("exec", cmd))
                raise RuntimeError("docker daemon hiccup")
            return super().exec_run(cmd, workdir=workdir, stdout=stdout, stderr=stderr)

    env = Tbench2DockerEnvironment()
    container = _ExplodingContainer()
    env._container = container
    env._task_dir = task

    with pytest.raises(RuntimeError, match="hiccup"):
        env._evaluate_docker()

    assert container.events[-1][0] == "exec"
    assert "rm -rf /task/tests" in container.events[-1][1]


def test_evaluate_docker_missing_tests_scores_zero(tmp_path: Path):
    task = tmp_path / "no-tests-task"
    task.mkdir()
    env = Tbench2DockerEnvironment()
    container = _FakeContainer()
    env._container = container
    env._task_dir = task

    output, reward, info = env._evaluate_docker()

    assert reward == 0.0
    assert container.events == []


@pytest.mark.skipif(camel is None, reason="camel-ai not installed")
@pytest.mark.skipif(
    os.environ.get("TB2_ENABLE_TESTS", "0") != "1",
    reason="TB2_ENABLE_TESTS not enabled",
)
def test_tbench2_env_smoke():
    env = Tbench2Environment(tasks_dir=os.environ.get("TB2_TASKS_DIR"))
    obs = env.reset(task_id=os.environ.get("TB2_TASK_ID", "headless-terminal"))
    assert obs.instruction

    result = env.step(Tbench2Action(action_type="exec", command="pwd"))
    assert result.success
    assert result.output

    env.step(Tbench2Action(action_type="close"))
    env.close()
