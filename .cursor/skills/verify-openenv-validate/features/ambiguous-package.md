# Ambiguous package

`openenv validate` refuses `tests/fixtures/validation/ambiguous_package` because it carries two well-known files (`openenv.yaml` and `task.toml`). It prints a signature error on stderr and exits 2 without guessing a format.

## Sub-features

- `ambiguous-refuse` exits 2.
- `ambiguous-stderr` says the package is ambiguous.
- `ambiguous-no-report` does not print a `Verdict:` line or a `ValidationReport` JSON object.

## How to get to it (user POV)

- Run `openenv validate tests/fixtures/validation/ambiguous_package --level static --skip-build`.

## Driving it with validate-run

Preconditions:

- Doctor exited 0.
- The fixture directory contains both `openenv.yaml` and `task.toml`.

- **Signature refusal.** Run `EXPECTED_EXIT=2 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" ambiguous-package -- validate tests/fixtures/validation/ambiguous_package --level static --skip-build`. Exit code `2`. Stderr contains `ambiguous` (case-insensitive). Stdout has no `Verdict:` and is not parseable as `ValidationReport`.
- **Proof.** Keep `$EVIDENCE_ROOT/ambiguous-package/{command,stdout,stderr,exit_code}.txt`.

## Gotchas

- The error is on **stderr**. Checking only stdout will miss it.
- Do not "fix" the fixture by deleting `task.toml` to make validate pass. Ambiguity is the feature.
- Harbor `task.toml` is not parsed in slice 1. Ambiguity is detected from well-known filenames before any Harbor parser exists.
- Exit 1 with a `static.manifest` FAIL would mean the CLI picked a format. That is a contract break.
