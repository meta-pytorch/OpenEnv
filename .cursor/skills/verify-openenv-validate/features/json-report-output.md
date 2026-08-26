# JSON report output

`--json` prints the validation report as schema-versioned JSON on stdout. `--output <file>` writes that JSON to disk. An author can pin `policy_version` and re-read the file as `ValidationReport`.

## Sub-features

- `json-stdout` replaces the human summary with JSON when `--json` is set.
- `json-file` writes JSON to `--output` even without `--json`.
- `json-schema` round-trips through `ValidationReport.model_validate_json`.

## How to get to it (user POV)

- Run `openenv validate envs/echo_env --level static --skip-build --json`.
- Run `openenv validate envs/echo_env --level static --skip-build --output report.json`.
- Combine `--json --output report.json` to print and write the same payload.

## Driving it with validate-run

Preconditions:

- Doctor exited 0.
- Echo env static validate is reachable (same package as that feature).

- **JSON stdout.** Run `EXPECTED_EXIT=0 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" json-report-stdout -- validate envs/echo_env --level static --skip-build --json`. Exit code `0`. `check-report.py` on `$EVIDENCE_ROOT/json-report-stdout/stdout.txt` prints `verdict=pass` `static.manifest=pass` `schema=1`.
- **Output file only.** Run `EXPECTED_EXIT=0 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" json-report-file -- validate envs/echo_env --level static --skip-build --output "$SCRATCH_ROOT/report.json"`. Exit code `0`. Stdout contains `Verdict: PASS` and does not start with `{`. `check-report.py "$SCRATCH_ROOT/report.json"` succeeds. Copy the file to `$EVIDENCE_ROOT/json-report-file/report.json` if needed.
- **Both.** Run `EXPECTED_EXIT=0 bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" json-report-both -- validate envs/echo_env --level static --skip-build --json --output "$SCRATCH_ROOT/report-both.json"`. Exit code `0`. Stdout JSON and the file both pass `check-report.py` with the same `verdict` and `check_id`s.
- **Proof.** Keep the three artifact directories. File proof is the bytes on disk, not the helper's return code alone.

## Gotchas

- Human mode plus `--output` still prints the text report. Do not require stdout to be JSON unless `--json` was passed.
- `--json` on a FAIL package still prints JSON and exits 1. Schema validity does not imply `verdict=pass`.
- Do not pretty-print or re-serialize the report before `check-report.py`. Validate the CLI's bytes.
