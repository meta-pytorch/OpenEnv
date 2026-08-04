"""Parser for the served OpenEnv format (`openenv.yaml`)."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..manifest import ManifestError, NormalizedManifest
from ..types import SignatureKind

VALIDATION_BLOCK_REMEDIATION = (
    "add a `validation:` block to openenv.yaml declaring reward (range, "
    "oracle_tolerance, floor_margin), resources (cpu, memory_mb, disk_mb, "
    "episode_timeout_s), capabilities (verifier, oracle, ...), and types (tags); "
    "see the manifest schema at src/openenv/validation/schemas/manifest.schema.json"
)


class OpenEnvYamlParser:
    """
    Parses `openenv.yaml` into the normalized manifest.

    Pure read: never imports or executes package code. Top-level `name`/`version`
    identify the environment; everything validation-specific lives under the
    `validation:` block, which maps 1:1 onto [`~openenv.validation.NormalizedManifest`]
    sections.
    """

    signature = SignatureKind.OPENENV_SERVED

    def parse(self, package_root: Path) -> NormalizedManifest:
        """
        Parse a served-environment package.

        Args:
            package_root (`Path`):
                Directory containing `openenv.yaml`.

        Returns:
            [`~openenv.validation.NormalizedManifest`]: the normalized manifest.
        """
        source = package_root / "openenv.yaml"
        try:
            raw = yaml.safe_load(source.read_text())
        except yaml.YAMLError as exc:
            raise ManifestError([f"openenv.yaml is not valid YAML: {exc}"]) from exc
        if not isinstance(raw, dict):
            raise ManifestError(["openenv.yaml must be a YAML mapping"])

        validation = raw.get("validation")
        if validation is None:
            raise ManifestError(
                ["openenv.yaml has no `validation:` block"],
                remediation=VALIDATION_BLOCK_REMEDIATION,
            )
        if not isinstance(validation, dict):
            raise ManifestError(["`validation:` must be a mapping"])

        data: dict = {
            "manifest_schema_version": "1",
            "signature": SignatureKind.OPENENV_SERVED,
            "version": raw.get("version"),
            "judge": validation.get("judge"),
            "task_distribution": validation.get("task_distribution"),
        }
        # Required sections are omitted when absent so schema errors read as
        # "field required" rather than "not a valid dictionary"; network is
        # omitted when absent so the manifest default (mode public) applies.
        for key, value in (
            ("name", raw.get("name")),
            ("reward", validation.get("reward")),
            ("resources", validation.get("resources")),
            ("capabilities", validation.get("capabilities")),
            ("types", validation.get("types")),
            ("network", validation.get("network")),
        ):
            if value is not None:
                data[key] = value

        try:
            return NormalizedManifest.model_validate(data)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(part) for part in err['loc']) or 'manifest'}: {err['msg']}"
                for err in exc.errors()
            ]
            raise ManifestError(
                errors, remediation=VALIDATION_BLOCK_REMEDIATION
            ) from exc
