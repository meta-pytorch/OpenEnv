"""Contract tests for the core validation enums."""

from openenv.validation.signature import WELL_KNOWN_FILES
from openenv.validation.types import (
    CheckStatus,
    Lane,
    Level,
    Severity,
    SignatureKind,
    Verdict,
)


def test_levels_are_ordered_by_cost():
    assert Level.STATIC < Level.RUNTIME < Level.SEMANTIC < Level.STATISTICAL


def test_grader_statuses_and_policy_severities_are_disjoint_vocabularies():
    # Graders emit CheckStatus; only the policy assigns Severity. SKIP/ERROR belong
    # to graders alone; ADVISORY belongs to the policy alone.
    statuses = {s.value for s in CheckStatus}
    severities = {s.value for s in Severity}
    assert "skip" in statuses and "skip" not in severities
    assert "error" in statuses and "error" not in severities
    assert "advisory" in severities and "advisory" not in statuses


def test_verdicts():
    assert {v.value for v in Verdict} == {"pass", "warn", "fail"}


def test_lanes():
    assert {lane.value for lane in Lane} == {"local", "hub"}


def test_well_known_files_track_implemented_parsers_only():
    # The detection table only lists formats this build can parse; entries are
    # added alongside their parsers. Every entry's filename is its enum value.
    assert set(WELL_KNOWN_FILES) <= set(SignatureKind)
    for kind, filename in WELL_KNOWN_FILES.items():
        assert kind.value == filename
