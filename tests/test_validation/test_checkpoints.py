"""Slice checkpoints, accreted permanently (RFC 008 structure §10).

A slice is not done until its checkpoint passes AND every earlier checkpoint still
passes. Each slice appends its checkpoint here so regressions are structural, not
procedural.
"""

import pytest
from conftest import (
    EXPECTED_POLICY,
    INVALID_MANIFEST_FIXTURES,
    load_fixture_manifest,
    VALID_MANIFEST_FIXTURES,
)
from openenv.validation.manifest import NormalizedManifest
from openenv.validation.policy import load_policy
from pydantic import ValidationError


def test_checkpoint_0_contracts_are_executable():
    """Slice 0: every contract is exercised the day it lands.

    The full evidence is this suite being green (schema round-trips, policy
    completeness, protocol conformance, schema sync). This checkpoint pins the
    load-bearing composite: fixtures validate or fail as intended against the
    committed schema, and the committed policy covers the complete check-id
    namespace including reserved operator ids.
    """
    for name in VALID_MANIFEST_FIXTURES:
        NormalizedManifest.model_validate(load_fixture_manifest(name))
    for name in INVALID_MANIFEST_FIXTURES:
        with pytest.raises(ValidationError):
            NormalizedManifest.model_validate(load_fixture_manifest(name))

    policy = load_policy("v1")
    assert {e.check_id for e in policy.entries} == set(EXPECTED_POLICY)
