"""Shared fixtures and canonical expectations for the validation contract tests."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "validation"

# Fixture packages whose normalized manifest must validate.
VALID_MANIFEST_FIXTURES = [
    "served_min_pass",
    "empty_solution_max_reward",
    "no_oracle",
    "leaky_observation",
    "nondeterministic",
    "leaky_egress",
    "harbor_task_min",
    "posttrain_task_min",
]

# Fixture packages whose normalized manifest must FAIL validation.
INVALID_MANIFEST_FIXTURES = [
    "broken_manifest",
    "unpinned_judge",
]


def load_fixture_manifest(name: str) -> dict:
    """Load a fixture package's normalized manifest JSON."""
    return json.loads((FIXTURES / name / "normalized_manifest.json").read_text())
