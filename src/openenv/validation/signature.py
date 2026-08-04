"""Signature detection contracts: well-known files, never a guess.

`detect_signature` itself lands with the walking skeleton (slice 1); this module ships
the detection rules and error contract so parsers, registries, and the CLI exit-code
contract can be written against them.
"""

from .types import SignatureKind

WELL_KNOWN_FILES: dict[SignatureKind, str] = {}
"""Signature detection table: the formats THIS build can parse.

Entries are added alongside their parsers (`openenv.yaml` with the served-env
parser, `task.toml` with the Harbor parser, `task.md` — frontmatter required —
with the PostTrain parser). A format absent from this table is refused as
unrecognized: validation never claims support for a format it cannot parse.
"""

UNSUPPORTED_CATEGORIES: dict[str, str] = {
    "hosted-verifier": (
        "the verifier calls an external service; the run cannot be hermetic and "
        "verifier portability is unmeasurable locally"
    ),
    "multi-agent": (
        "multi-agent scenes need an orchestration harness local validation does not define"
    ),
    "simulated-user": (
        "requires a user-simulator policy; no local reference simulator exists"
    ),
    "parser-not-implemented": (
        "defensive registry contract: a signature kind reached dispatch without a "
        "registered parser; unreachable when detection tracks implemented parsers"
    ),
}
"""The RFC 008 unsupported-categories list. Detection produces exit code 2, never a guess."""


class SignatureError(Exception):
    """
    Ambiguous or unrecognized package: zero or two+ well-known files matched.

    Maps to CLI exit code 2.
    """


class UnsupportedPackageError(Exception):
    """
    A recognized package in a category local validation does not attempt.

    Raised at parse time where detectable. Maps to CLI exit code 2.

    Attributes:
        category (`str`):
            A key of [`~openenv.validation.signature.UNSUPPORTED_CATEGORIES`].
        reason (`str`):
            Human-readable explanation for this package.
    """

    def __init__(self, category: str, reason: str):
        if category not in UNSUPPORTED_CATEGORIES:
            raise ValueError(f"unknown unsupported-package category: {category!r}")
        super().__init__(f"unsupported package ({category}): {reason}")
        self.category = category
        self.reason = reason
