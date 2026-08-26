# Broken manifest

`openenv validate` on `tests/fixtures/validation/broken_manifest` treats an invalid `validation:` block as a graded failure: it prints a report that FAILs `static.manifest` and exits 1.

## Sub-features

- `broken-human` prints `FAIL  static.manifest` and `Verdict: FAIL`.
- `broken-exit` exits 1 (FAIL), not 2 (signature) or 3 (internal).
- `broken-json` emits a schema-valid report with `verdict` `fail` and `manifest` null or a failed check naming the schema problem.

## How to get to it (user POV)

- Run `openenv validate tests/fixtures/validation/broken_manifest --level static --skip-build`.
- Add `--json` when asserting the report object.

## Driving it with validate-run

Preconditions:

- Doctor exited 0.
- `tests/fixtures/validation/broken_manifest/openenv.yaml` exists (reward range `[1.0, 0.0]`).

- **Human FAIL.** Run `EXPECTED_EXIT=1 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" broken-manifest -- validate tests/fixtures/validation/broken_manifest --level static --skip-build`. Exit code `1`. Stdout contains `static.manifest` and `Verdict: FAIL`.
- **JSON FAIL.** Run `EXPECTED_EXIT=1 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" broken-manifest-json -- validate tests/fixtures/validation/broken_manifest --level static --skip-build --json`. Exit code `1`. `check-report.py` on stdout accepts the JSON and prints `verdict=fail` and `static.manifest=fail`.
- **Proof.** Keep both artifact directories. This is a successful verification of a failing package: exit 1 is the expected user-visible result.

## Gotchas

- Do not treat exit 1 as "the CLI is broken." The CLI worked; the package failed the bar.
- A `ManifestError` is graded, not a `SignatureError`. Exit 2 here is a regression.
- Remediation text distinguishes a missing `validation:` block from an invalid one. This fixture is invalid (range inverted), not missing.
- `--json` still exits 1. Capture stdout anyway; the report is the proof.
