# Unrecognized package

`openenv validate` refuses a directory with no well-known package file. It prints an unrecognized-package error on stderr and exits 2.

## Sub-features

- `unrecognized-refuse` exits 2 on `tests/fixtures/validation/unrecognized_package`.
- `unrecognized-stderr` says the package is unrecognized.
- `unrecognized-no-guess` does not invent an `openenv.yaml` or emit `Verdict:`.

## How to get to it (user POV)

- Run `openenv validate tests/fixtures/validation/unrecognized_package --level static --skip-build`.

## Driving it with validate-run

Preconditions:

- Doctor exited 0.
- The fixture directory has no `openenv.yaml`, `task.toml`, or `task.md`.

- **Signature refusal.** Run `EXPECTED_EXIT=2 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" unrecognized-package -- validate tests/fixtures/validation/unrecognized_package --level static --skip-build`. Exit code `2`. Stderr contains `unrecognized` (case-insensitive). Stdout has no `Verdict:`.
- **Proof.** Keep `$EVIDENCE_ROOT/unrecognized-package/{command,stdout,stderr,exit_code}.txt`.

## Gotchas

- A Harbor-only `task.toml` package (`tests/fixtures/validation/harbor_task_min`) also exits 2 in slice 1 because no Harbor parser ships yet. That is "unsupported format", not this fixture. Do not use it as a stand-in for unrecognized.
- A missing directory (path does not exist) also exits 2 (`not a package directory`). That is a different message. Use the committed empty-ish fixture.
