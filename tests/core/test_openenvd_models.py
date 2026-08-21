# SPDX-License-Identifier: BSD-3-Clause

"""Tests for openenvd task models."""

import pytest
from openenv.core.openenvd.models import RestartPolicy, TaskSpec, TaskState, TaskStatus
from pydantic import ValidationError


class TestTaskSpec:
    def test_minimal_spec(self):
        spec = TaskSpec(name="observer", argv=["sleep", "30"])
        assert spec.name == "observer"
        assert spec.argv == ["sleep", "30"]
        assert spec.env == {}
        assert spec.cwd is None
        assert spec.uid is None
        assert spec.gid is None
        assert spec.network_isolated is False
        assert spec.restart_policy == RestartPolicy.NEVER
        assert spec.max_retries == 3
        assert spec.stop_grace_s == 5.0

    def test_empty_argv_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec(name="x", argv=[])

    def test_invalid_name_rejected(self):
        for bad in ["", "has space", "-leading", "UPPER", "a/b"]:
            with pytest.raises(ValidationError):
                TaskSpec(name=bad, argv=["true"])

    @pytest.mark.parametrize(
        "policy", ["never", "on_failure", "always", RestartPolicy.ALWAYS]
    )
    def test_restart_policy_accepted(self, policy):
        spec = TaskSpec(name="x", argv=["true"], restart_policy=policy)
        assert spec.restart_policy == RestartPolicy(policy)

    def test_invalid_restart_policy_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec(name="x", argv=["true"], restart_policy="sometimes")

    def test_negative_values_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec(name="x", argv=["true"], max_retries=-1)
        with pytest.raises(ValidationError):
            TaskSpec(name="x", argv=["true"], stop_grace_s=-1.0)


class TestTaskStatus:
    def test_defaults(self):
        status = TaskStatus(name="obs", state=TaskState.RUNNING)
        assert status.pid is None
        assert status.exit_code is None
        assert status.restarts == 0

    def test_full_status(self):
        status = TaskStatus(
            name="obs",
            state=TaskState.EXITED,
            pid=123,
            exit_code=0,
            restarts=2,
        )
        assert status.exit_code == 0
