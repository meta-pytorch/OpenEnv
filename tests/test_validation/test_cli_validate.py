"""CLI contract: one command, exit codes 0/1/2/3, schema-valid JSON reports."""

import pytest
from conftest import FIXTURES
from openenv.cli.__main__ import app
from openenv.validation.report import ValidationReport
from typer.testing import CliRunner

runner = CliRunner()


def _validate(*args):
    return runner.invoke(app, ["validate", *args])


def test_valid_package_exits_zero():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build"
    )
    assert result.exit_code == 0, result.output
    assert "Verdict: PASS" in result.output


def test_json_report_is_schema_valid():
    result = _validate(
        str(FIXTURES / "served_min_pass"), "--level", "static", "--skip-build", "--json"
    )
    assert result.exit_code == 0, result.output
    report = ValidationReport.model_validate_json(result.output)
    assert report.verdict.value == "pass"


def test_output_writes_the_json_report(tmp_path):
    out = tmp_path / "report.json"
    result = _validate(
        str(FIXTURES / "served_min_pass"),
        "--level",
        "static",
        "--skip-build",
        "--output",
        str(out),
    )
    assert result.exit_code == 0, result.output
    ValidationReport.model_validate_json(out.read_text())


def test_failing_manifest_exits_one():
    result = _validate(
        str(FIXTURES / "broken_manifest"), "--level", "static", "--skip-build"
    )
    assert result.exit_code == 1, result.output
    assert "static.manifest" in result.output


def test_unrecognized_package_exits_two():
    result = _validate(
        str(FIXTURES / "unrecognized_package"), "--level", "static", "--skip-build"
    )
    assert result.exit_code == 2, result.output
    assert "unrecognized" in result.output


def test_format_without_a_parser_exits_two():
    # Harbor packages are refused as unrecognized until the Harbor parser lands.
    result = _validate(
        str(FIXTURES / "harbor_task_min"), "--level", "static", "--skip-build"
    )
    assert result.exit_code == 2, result.output
    assert "unrecognized" in result.output


def test_nonexistent_path_exits_two(tmp_path):
    result = _validate(str(tmp_path / "nope"))
    assert result.exit_code == 2, result.output


def test_unknown_level_is_an_internal_error():
    result = _validate(str(FIXTURES / "served_min_pass"), "--level", "cosmic")
    assert result.exit_code == 3, result.output


@pytest.mark.parametrize("fixture", ["served_min_pass", "broken_manifest"])
def test_unpinned_judge_fails_from_slice_1_onward(fixture):
    # Sanity anchor for the F2 note: unpinned_judge FAILs static.manifest already.
    result = _validate(
        str(FIXTURES / "unpinned_judge"), "--level", "static", "--skip-build"
    )
    assert result.exit_code == 1
    assert "judge pin" in result.output
