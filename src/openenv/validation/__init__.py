"""RFC 008 validation contracts: types and the normalized manifest.

This slice ships the enums and [`~openenv.validation.manifest.NormalizedManifest`].
Graders, when they land, read the manifest and never the signature. Detection,
parsers, the report schema, and the severity policy land in later slices.
"""

from .manifest import (
    CapabilitiesSpec,
    JudgePin,
    NetworkPolicy,
    NormalizedManifest,
    OracleDeclaration,
    ResourceDeclaration,
    RewardDeclaration,
    TaskDistributionPin,
    TypeSpec,
    VerifierBinding,
)
from .types import (
    CheckStatus,
    Lane,
    Level,
    ProviderCapability,
    Severity,
    SignatureKind,
    Verdict,
)

__all__ = [
    "CapabilitiesSpec",
    "CheckStatus",
    "JudgePin",
    "Lane",
    "Level",
    "NetworkPolicy",
    "NormalizedManifest",
    "OracleDeclaration",
    "ProviderCapability",
    "ResourceDeclaration",
    "RewardDeclaration",
    "Severity",
    "SignatureKind",
    "TaskDistributionPin",
    "TypeSpec",
    "Verdict",
    "VerifierBinding",
]
