# Echo env static validate

`openenv validate` on `envs/echo_env` at the static level accepts the committed `openenv.yaml` `validation:` block, prints a PASS report, and exits 0.

## Sub-features

- `echo-human` prints the text report with `PASS  static.manifest` and `Verdict: PASS`.
- `echo-json` prints a schema-valid `ValidationReport` whose `verdict` is `pass`.
- `echo-output` writes the same report to `--output` while still printing the human summary.

## How to get to it (user POV)

- From the repo root, run `openenv validate envs/echo_env --level static --skip-build`.
- Add `--json` for stdout JSON.
- Add `--output <file>` to write the JSON report to disk.

## Driving it with validate-run

Preconditions:

- Doctor exited 0.
- `envs/echo_env/openenv.yaml` contains `name: echo_env` and a `validation:` block.
- `$SCRATCH_ROOT` exists and is writable.

- **Human entry.** Validate echo_env as an author would. Run `bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" echo-env-static-validate -- validate envs/echo_env --level static --skip-build`. Exit code `0`. Stdout contains `Validation report for`, `PASS  static.manifest`, `signature: openenv.yaml`, `levels run: static`, and `Verdict: PASS`. Stderr is empty or lacks `Error:`.
- **JSON entry.** Repeat with `--json`. Run `EXPECTED_EXIT=0 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" echo-env-static-validate-json -- validate envs/echo_env --level static --skip-build --json`. Exit code `0`. Stdout is JSON. Run `uv run python "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/check-report.py" "$EVIDENCE_ROOT/echo-env-static-validate-json/stdout.txt"`. It prints `verdict=pass` and `static.manifest=pass`.
- **Output file.** Write a report beside the human printout. Run `EXPECTED_EXIT=0 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" echo-env-static-validate-output -- validate envs/echo_env --level static --skip-build --output "$SCRATCH_ROOT/echo-env-report.json"`. Exit code `0`. Stdout is the human report (`Verdict: PASS`), not JSON. `$SCRATCH_ROOT/echo-env-report.json` exists. Copy it to `$EVIDENCE_ROOT/echo-env-static-validate-output/report.json` if the helper did not. `check-report.py` on that file prints `verdict=pass` and `static.manifest=pass`.
- **Proof.** Keep `$EVIDENCE_ROOT/echo-env-static-validate/`, `echo-env-static-validate-json/`, and `echo-env-static-validate-output/`. Each directory has `command.txt`, `stdout.txt`, `stderr.txt`, and `exit_code.txt`.

## Gotchas

- Default `--level` is `semantic`, not `static`. Slice 1 has no semantic graders; still pass `--level static` so the proof matches the checkpoint and `levels_run` is `[static]`.
- `--skip-build` does not skip a build today. Pass it anyway; the user path in the CLI help names it.
- `source_digest` and `duration_s` change every run. Do not pin them.
- `manifest` is present and non-null on this path. A null `manifest` means parse failed — that is a FAIL package, not echo_env.
- Do not use `PYTHONPATH=src:envs` as a substitute for `uv run`. The installed package is enough for `openenv validate`.
- A green `pytest tests/test_validation/test_checkpoints.py` is not this feature's proof. Drive the CLI.
