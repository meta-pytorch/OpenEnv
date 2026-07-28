# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Harbor environment.

Everything here is hermetic: no Docker, no network, no API keys. The bundled
`fix-sum-bug` example task doubles as the end-to-end fixture, so these tests also
guard the claim that a real Harbor task directory runs unmodified.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from harbor_env import HarborAction, HarborEnv, HarborObservation, HarborState
from harbor_env.server import (
    HarborEnvironment,
    HarborTask,
    LocalSandbox,
    SandboxError,
    TaskCatalog,
    TaskFormatError,
)
from harbor_env.server.reward import read_reward
from harbor_env.server.sandbox import expand_env_refs, resolve_within, SandboxPaths
from harbor_env.server.task import BUNDLED_TASKS_DIR, resolve_task_source
from openenv.core.env_server.serialization import serialize_observation


EXAMPLE_TASK_ID = "fix-sum-bug"

#: The bundled task's grader runs five checks; the shipped buggy `stats.py`
#: passes three of them.
BUGGY_REWARD = 3 / 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_task(
    root: Path,
    task_id: str,
    *,
    config: str | None = None,
    instruction: str = "do the thing",
    test_script: str | None = None,
    solve_script: str | None = None,
    seed_files: dict[str, str] | None = None,
    dockerfile: str | None = None,
) -> Path:
    """Materialize a minimal Harbor task directory for a test."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("task.toml").write_text(
        config
        if config is not None
        else textwrap.dedent(
            f"""
            schema_version = "1.4"

            [task]
            name = "tests/{task_id}"
            description = "fixture"
            """
        ).strip(),
        encoding="utf-8",
    )
    task_dir.joinpath("instruction.md").write_text(instruction, encoding="utf-8")

    if test_script is not None:
        tests = task_dir / "tests"
        tests.mkdir(exist_ok=True)
        tests.joinpath("test.sh").write_text(test_script, encoding="utf-8")
    if solve_script is not None:
        solution = task_dir / "solution"
        solution.mkdir(exist_ok=True)
        solution.joinpath("solve.sh").write_text(solve_script, encoding="utf-8")
    for name, content in (seed_files or {}).items():
        target = task_dir / "environment" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if dockerfile is not None:
        environment = task_dir / "environment"
        environment.mkdir(exist_ok=True)
        environment.joinpath("Dockerfile").write_text(dockerfile, encoding="utf-8")
    return task_dir


def reader(files: dict[str, str]):
    """Build the `read_text` callable that `read_reward` expects."""
    return lambda name: files.get(name)


# ---------------------------------------------------------------------------
# task.toml parsing
# ---------------------------------------------------------------------------


def test_load_reads_harbor_schema(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "demo",
        config=textwrap.dedent(
            """
            schema_version = "1.4"

            [task]
            name = "acme/demo"
            description = "a demo"

            [metadata]
            difficulty = "easy"

            [environment]
            docker_image = "python:3.12-slim"

            [agent]
            timeout_sec = 60.0

            [verifier]
            timeout_sec = 30.0
            """
        ).strip(),
        instruction="# Demo\n",
    )

    task = HarborTask.load(task_dir)

    assert task.name == "acme/demo"
    assert task.description == "a demo"
    assert task.schema_version == "1.4"
    assert task.instruction == "# Demo\n"
    assert task.metadata["difficulty"] == "easy"
    assert task.docker_image == "python:3.12-slim"
    assert task.agent_timeout_s == 60.0
    assert task.verifier_timeout_s == 30.0


def test_load_reads_repo2rlenv_version_key(tmp_path: Path) -> None:
    """Repo2RLEnv writes `version` where Harbor writes `schema_version`."""
    task_dir = write_task(
        tmp_path,
        "pallets__click-2951",
        config=textwrap.dedent(
            """
            version = "1.0"

            [task]
            name = "acme/pallets__click-2951"
            description = "Fix the thing"

            [metadata.repo2env]
            pipeline = "pr_diff"
            reward_kinds = ["diff_similarity"]
            """
        ).strip(),
    )

    task = HarborTask.load(task_dir)

    assert task.schema_version == "1.0"
    assert task.metadata["repo2env"]["pipeline"] == "pr_diff"
    # Absent timeout tables fall back to Harbor's documented defaults.
    assert task.verifier_timeout_s == 300.0
    assert task.agent_timeout_s == 1800.0


def test_load_rejects_non_task_directory(tmp_path: Path) -> None:
    with pytest.raises(TaskFormatError, match="no task.toml"):
        HarborTask.load(tmp_path)


def test_load_rejects_windows_tasks(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "win",
        config='schema_version = "1.4"\n\n[environment]\nos = "windows"\n',
    )
    with pytest.raises(TaskFormatError, match="only supports Linux"):
        HarborTask.load(task_dir)


def test_instruction_falls_back_to_description(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "terse")
    (task_dir / "instruction.md").unlink()
    assert HarborTask.load(task_dir).instruction == "fixture"


# ---------------------------------------------------------------------------
# self-contained vs image-backed tasks
# ---------------------------------------------------------------------------


def test_seed_files_make_a_task_self_contained(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "seeded", seed_files={"app.py": "x = 1\n"})
    task = HarborTask.load(task_dir)

    assert [p.name for p in task.seed_files] == ["app.py"]
    assert task.needs_image is False


def test_dockerfile_suppresses_seed_upload(tmp_path: Path) -> None:
    """Harbor only uploads environment/ when it holds no build recipe."""
    task_dir = write_task(
        tmp_path,
        "built",
        seed_files={"app.py": "x = 1\n"},
        dockerfile="FROM python:3.12-slim\n",
    )
    task = HarborTask.load(task_dir)

    assert task.seed_files == ()
    assert task.needs_image is True


def test_docker_image_alone_needs_an_image(tmp_path: Path) -> None:
    task_dir = write_task(
        tmp_path,
        "prebuilt",
        config='schema_version = "1.4"\n\n[environment]\ndocker_image = "ghcr.io/acme/x:1"\n',
    )
    assert HarborTask.load(task_dir).needs_image is True


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_catalog_finds_flat_and_nested_tasks(tmp_path: Path) -> None:
    write_task(tmp_path, "alpha")
    write_task(tmp_path / "tasks", "beta")  # the layout `repo2rlenv push` writes

    catalog = TaskCatalog(tmp_path)

    assert catalog.task_ids() == ["alpha", "tasks/beta"]
    assert catalog.get("tasks/beta").task_id == "tasks/beta"
    # A bare directory name resolves when it is unambiguous.
    assert catalog.get("beta").task_id == "tasks/beta"


def test_catalog_accepts_a_single_task_directory(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, "solo")
    assert TaskCatalog(task_dir).task_ids() == ["solo"]


def test_catalog_reports_ambiguous_and_unknown_ids(tmp_path: Path) -> None:
    write_task(tmp_path / "a", "dup")
    write_task(tmp_path / "b", "dup")
    catalog = TaskCatalog(tmp_path)

    with pytest.raises(KeyError, match="ambiguous"):
        catalog.get("dup")
    with pytest.raises(KeyError, match="unknown task"):
        catalog.get("nope")


def test_catalog_caches_its_scan_until_refreshed(tmp_path: Path) -> None:
    write_task(tmp_path, "first")
    catalog = TaskCatalog(tmp_path)
    assert catalog.task_ids() == ["first"]

    write_task(tmp_path, "second")
    assert catalog.task_ids() == ["first"]  # served task sets are fixed

    catalog.refresh()
    assert catalog.task_ids() == ["first", "second"]


def test_resolve_task_source_defaults_to_bundled_tasks() -> None:
    assert resolve_task_source(None) == BUNDLED_TASKS_DIR
    assert resolve_task_source("  ") == BUNDLED_TASKS_DIR


def test_resolve_task_source_rejects_nonsense() -> None:
    with pytest.raises(FileNotFoundError, match="neither an existing directory"):
        resolve_task_source("./definitely/not/here")


# ---------------------------------------------------------------------------
# reward contract
# ---------------------------------------------------------------------------


def test_reward_json_takes_precedence_over_reward_txt() -> None:
    report = read_reward(
        reader(
            {"reward.json": '{"reward": 0.75, "f2p_rate": 0.5}', "reward.txt": "0.1"}
        )
    )

    assert report.value == 0.75
    assert report.source == "reward.json"
    assert report.metrics == {"reward": 0.75, "f2p_rate": 0.5}
    assert report.graded is True


def test_reward_json_with_a_single_metric_is_unambiguous() -> None:
    report = read_reward(reader({"reward.json": '{"accuracy": 0.5}'}))
    assert report.value == 0.5
    assert report.source == "reward.json"


def test_reward_json_without_a_primary_metric_falls_back_to_txt() -> None:
    report = read_reward(
        reader(
            {
                "reward.json": '{"accuracy": 0.5, "runtime_sec": 2.0}',
                "reward.txt": "0.9",
            }
        )
    )

    assert report.value == 0.9
    assert report.source == "reward.txt"
    assert report.metrics == {"accuracy": 0.5, "runtime_sec": 2.0}
    assert any("no 'reward' key" in e for e in report.errors)


def test_reward_txt_alone_is_honoured() -> None:
    report = read_reward(reader({"reward.txt": "1\n"}))
    assert report.value == 1.0
    assert report.source == "reward.txt"


def test_missing_reward_files_are_not_synthesized_into_zero() -> None:
    report = read_reward(reader({}))

    assert report.value is None
    assert report.graded is False
    assert report.source == "missing"


def test_malformed_reward_files_are_reported_not_guessed() -> None:
    report = read_reward(reader({"reward.json": "{oops", "reward.txt": "not-a-number"}))

    assert report.value is None
    assert len(report.errors) == 2


def test_repo2rlenv_details_sidecar_is_surfaced_but_never_scored() -> None:
    details = {"reward": 0.83, "f2p_passed": 2, "components": {"similarity": 0.4}}
    report = read_reward(
        reader({"reward-details.json": json.dumps(details), "reward.txt": "0.83"})
    )

    assert report.value == 0.83
    assert report.source == "reward.txt"
    assert report.details["components"]["similarity"] == 0.4


# ---------------------------------------------------------------------------
# sandbox primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relative", ["a.py", "pkg/mod.py", "./pkg/../a.py"])
def test_resolve_within_accepts_paths_inside_the_workdir(relative: str) -> None:
    assert resolve_within("/work", relative).startswith("/work")


@pytest.mark.parametrize(
    "relative", ["../tests/grade.py", "../../etc/passwd", "/etc/passwd", "a/../../b"]
)
def test_resolve_within_rejects_escapes(relative: str) -> None:
    with pytest.raises(ValueError):
        resolve_within("/work", relative)


def test_sandbox_paths_expose_harbor_layout_as_env() -> None:
    env = SandboxPaths(workdir="/workspace").as_env()

    assert env["HARBOR_WORKDIR"] == "/workspace"
    assert env["HARBOR_TESTS_DIR"] == "/tests"
    assert env["HARBOR_LOGS_DIR"] == "/logs/verifier"


def test_expand_env_refs_resolves_and_blanks_missing_values(monkeypatch) -> None:
    monkeypatch.setenv("HARBOR_TEST_TOKEN", "s3cret")
    monkeypatch.delenv("HARBOR_TEST_ABSENT", raising=False)

    resolved = expand_env_refs(
        {"A": "${HARBOR_TEST_TOKEN}", "B": "$HARBOR_TEST_ABSENT"}
    )

    assert resolved == {"A": "s3cret", "B": ""}


def test_local_sandbox_refuses_image_backed_tasks(tmp_path: Path) -> None:
    task = HarborTask.load(
        write_task(tmp_path, "needs-image", dockerfile="FROM python:3.12-slim\n")
    )
    sandbox = LocalSandbox()
    try:
        with pytest.raises(SandboxError, match="HARBOR_MODE=docker"):
            sandbox.start(task)
    finally:
        sandbox.close()


def test_local_sandbox_cleans_up_its_root(tmp_path: Path) -> None:
    task = HarborTask.load(
        write_task(tmp_path, "seeded", seed_files={"a.py": "x = 1\n"})
    )
    sandbox = LocalSandbox()
    sandbox.start(task)
    root = sandbox.root

    assert (Path(sandbox.paths.workdir) / "a.py").is_file()
    sandbox.close()
    assert not root.exists()


# ---------------------------------------------------------------------------
# environment: the bundled example task, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    environment = HarborEnvironment()
    yield environment
    environment.close()


def test_reset_serves_the_task_instruction(env: HarborEnvironment) -> None:
    obs = env.reset(task_id=EXAMPLE_TASK_ID)

    assert obs.task_id == EXAMPLE_TASK_ID
    assert obs.task_name == "openenv/fix-sum-bug"
    assert obs.mode == "local"
    assert "stats.py" in obs.instruction
    assert obs.done is False
    assert env.state.available_tasks == [EXAMPLE_TASK_ID]


def test_seed_files_land_in_the_working_directory(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="read", path="stats.py"))

    assert obs.success is True
    assert "def total(values)" in obs.output


def test_verifier_reward_is_forwarded_verbatim(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="evaluate"))

    assert obs.reward == pytest.approx(BUGGY_REWARD)
    assert obs.done is True
    assert obs.info["reward_source"] == "reward.json"
    assert obs.info["reward_metrics"]["total_empty"] == 0.0
    assert env.state.evaluated is True
    assert env.state.reward == pytest.approx(BUGGY_REWARD)


def test_agent_edits_change_the_reward(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    fixed = textwrap.dedent(
        """
        def total(values):
            return sum(values)


        def mean(values):
            if not values:
                return 0.0
            return total(values) / len(values)
        """
    ).strip()
    write = env.step(HarborAction(action_type="write", path="stats.py", content=fixed))
    assert write.success is True

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(1.0)


def test_oracle_solution_earns_full_reward(env: HarborEnvironment) -> None:
    """`solve` is how a task set is validated: the oracle must score 1.0."""
    env.reset(task_id=EXAMPLE_TASK_ID)
    solved = env.step(HarborAction(action_type="solve"))
    assert solved.success is True

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(1.0)


def test_exec_reports_failures_without_ending_the_episode(
    env: HarborEnvironment,
) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    obs = env.step(HarborAction(action_type="exec", command="exit 3"))

    assert obs.success is False
    assert obs.exit_code == 3
    assert obs.done is False
    assert obs.reward is None


def test_agent_cannot_reach_the_verifier(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)

    escape = env.step(HarborAction(action_type="read", path="../tests/grade.py"))
    assert escape.success is False
    assert "escapes the working directory" in escape.error

    # tests/ is staged only when the verifier runs, so it is not there yet.
    listing = env.step(HarborAction(action_type="exec", command="ls"))
    assert listing.output.split() == ["stats.py"]


def test_planted_reward_files_are_discarded(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    env.step(
        HarborAction(
            action_type="exec",
            command='mkdir -p "$HARBOR_LOGS_DIR" && echo \'{"reward": 1.0}\' > "$HARBOR_LOGS_DIR/reward.json"',
        )
    )

    assert env.step(HarborAction(action_type="evaluate")).reward == pytest.approx(
        BUGGY_REWARD
    )


def test_unscorable_episode_reports_no_reward(tmp_path: Path) -> None:
    """A verifier that writes nothing must not be turned into a 0.0."""
    write_task(
        tmp_path,
        "silent",
        seed_files={"noop.txt": "hi\n"},
        test_script="#!/bin/bash\necho 'graded nothing'\n",
    )
    environment = HarborEnvironment(tasks=str(tmp_path))
    try:
        environment.reset(task_id="silent")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.reward is None
    assert obs.done is True
    assert obs.success is False
    assert "wrote no reward file" in obs.error
    assert obs.info["reward_source"] == "missing"


def test_missing_verifier_is_reported(tmp_path: Path) -> None:
    write_task(tmp_path, "ungraded", seed_files={"a.txt": "x\n"})
    environment = HarborEnvironment(tasks=str(tmp_path))
    try:
        environment.reset(task_id="ungraded")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.success is False
    assert "no tests/test.sh" in obs.error


def test_absolute_path_verifier_gets_an_actionable_hint(tmp_path: Path) -> None:
    write_task(
        tmp_path,
        "hardcoded",
        seed_files={"a.txt": "x\n"},
        test_script="#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n",
    )
    environment = HarborEnvironment(tasks=str(tmp_path))
    try:
        environment.reset(task_id="hardcoded")
        obs = environment.step(HarborAction(action_type="evaluate"))
    finally:
        environment.close()

    assert obs.reward is None
    assert "HARBOR_MODE=docker" in obs.error


def test_step_before_reset_is_reported(env: HarborEnvironment) -> None:
    obs = env.step(HarborAction(action_type="exec", command="ls"))

    assert obs.success is False
    assert "call reset() first" in obs.error


def test_reset_without_a_task_id_lists_the_options(tmp_path: Path) -> None:
    write_task(tmp_path, "one")
    write_task(tmp_path, "two")
    environment = HarborEnvironment(tasks=str(tmp_path))
    try:
        with pytest.raises(ValueError, match="needs a task_id"):
            environment.reset()
    finally:
        environment.close()


def test_reset_starts_a_clean_episode(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    env.step(HarborAction(action_type="write", path="scratch.txt", content="dirty"))
    first_workdir = env.state.workdir

    env.reset(task_id=EXAMPLE_TASK_ID)

    assert env.state.workdir != first_workdir
    assert env.state.step_count == 0
    assert env.step(HarborAction(action_type="exec", command="ls")).output.split() == [
        "stats.py"
    ]


# ---------------------------------------------------------------------------
# wire contract between client and server
# ---------------------------------------------------------------------------


def test_action_schema_rejects_unknown_action_types() -> None:
    with pytest.raises(ValueError):
        HarborAction(action_type="rm_rf")


def test_client_parses_what_the_server_serializes(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)
    observation = env.step(HarborAction(action_type="evaluate"))
    payload = serialize_observation(observation)

    result = HarborEnv(base_url="http://localhost:8000")._parse_result(payload)

    assert isinstance(result.observation, HarborObservation)
    assert result.reward == pytest.approx(BUGGY_REWARD)
    assert result.done is True
    assert result.observation.info["reward_source"] == "reward.json"


def test_client_parses_state(env: HarborEnvironment) -> None:
    env.reset(task_id=EXAMPLE_TASK_ID)

    state = HarborEnv(base_url="http://localhost:8000")._parse_state(
        env.state.model_dump()
    )

    assert isinstance(state, HarborState)
    assert state.task_id == EXAMPLE_TASK_ID
    assert state.mode == "local"


def test_client_step_payload_round_trips(tmp_path: Path) -> None:
    action = HarborAction(action_type="write", path="a.py", content="x = 1\n")

    payload = HarborEnv(base_url="http://localhost:8000")._step_payload(action)

    assert HarborAction.model_validate(payload) == action
