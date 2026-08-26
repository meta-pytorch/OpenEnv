---
name: verify-openenv-validate
description: "Drive the OpenEnv CLI `openenv validate` the way an author does: local package path, stdout/stderr, exit codes, and a schema-valid ValidationReport. Use when proving echo_env or fixture validation, checking RFC 008 slice-1 checkpoints, or verifying validate CLI behavior after validation-pipeline changes."
---

# Verify `openenv validate`

Short-lived CLI. There is no always-on server for local static validation. Launch means the checkout can run `openenv validate`; each drive is one isolated command that writes its own evidence.

Primary user surface: `uv run openenv validate <package>`. The equivalent spawn used by this repo's checkpoint tests is `uv run python -m openenv.cli validate <package>`. Prefer the user-facing `openenv` form unless a recipe names the module form.

`--url` probes a running FastAPI env (legacy runtime). That is a different surface; do not treat a live `echo_env` server as proof of local static validation.

## Launch

From the OpenEnv repo root (`/workspace` in this checkout):

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export EVIDENCE_ROOT="$REPO_ROOT/.cursor/skills/verify-openenv-validate/artifacts/$RUN_ID"
export SCRATCH_ROOT="/tmp/openenv-validate-verify-$RUN_ID"
mkdir -p "$EVIDENCE_ROOT" "$SCRATCH_ROOT"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO_ROOT"
```

Ready when this exits 0 and prints the validate help (must mention `--level` and `--skip-build`):

```bash
uv run openenv validate --help
```

If `uv` is missing, install it to `~/.local/bin` and re-run. If the import fails, `uv sync --all-extras` from `$REPO_ROOT` (core extras are enough for this CLI). Do not start `echo_env.server.app` for static recipes.

Teardown is Cleanup below. Launch does not leave a daemon.

## Doctor

Read-only. Run first whenever anything looks off:

```bash
bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/doctor.sh"
```

Doctor must print `ok` for each of: `uv` on PATH, `openenv validate --help` exits 0, `envs/echo_env/openenv.yaml` exists and contains a `validation:` block, and `ValidationReport` imports. Exit 0 means the instance is worth driving. Exit 1 means stop and fix; do not drive a half-imported CLI.

Two static validates may run side by side. They only read package trees and write reports under `$EVIDENCE_ROOT` / `$SCRATCH_ROOT`. Never write into `envs/echo_env/` or `tests/fixtures/validation/`.

## Drive

Read `features/README.md`, then the matching feature file. Follow that file's harness section literally.

Harness is the CLI helper (tmux optional; the helper already isolates stdout/stderr/exit):

```bash
bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/validate-run.sh" \
  <feature-id> -- \
  validate <package> [flags]
```

That runs `uv run openenv …` from `$REPO_ROOT`, writes `stdout.txt`, `stderr.txt`, `exit_code.txt`, `command.txt` under `$EVIDENCE_ROOT/<feature-id>/`, and copies `--output` reports into the same directory when the recipe names one.

Stable handles (assert these, not durations or digest strings):

- Human stdout: `Validation report for <path>`, `PASS  static.manifest`, `Verdict: PASS` / `Verdict: FAIL`
- JSON stdout or `--output` file: `report_schema_version` is `"1"`, `verdict` is `pass`/`fail`, a `results[]` entry has `check_id` `static.manifest`
- Exit codes: `0` PASS/WARN, `1` FAIL, `2` SignatureError / unrecognized / unsupported, `3` PolicyError / unknown `--level`
- Signature-error text goes to **stderr** (`ambiguous`, `unrecognized`), not stdout
- Schema-valid JSON means `ValidationReport.model_validate_json(...)` succeeds

Slice 1 only implements `static.manifest`. `--level static --skip-build` is the inner-loop contract. `--skip-build` is a no-op until build graders exist; still pass it so the user path matches the checkpoint.

## Evidence

Proof lives under `$EVIDENCE_ROOT` (see Launch). Cleanup must not delete it.

Standards:

- Drive the real CLI (`uv run openenv validate …`), not `CliRunner`, not `run_validation()` as a library call, not pytest as a substitute for the user path.
- Capture the command, stdout, stderr, and exit code for every drive. A passing pytest file is not this skill's proof.
- For PASS packages, also capture a JSON report (`--json` and/or `--output`) and validate it with `ValidationReport`.
- For FAIL packages, human stdout must name `static.manifest` and the process must exit 1.
- For signature refusals, stderr must name the reason and the process must exit 2. There is no report file.
- `--output PATH` proof is the file on disk, not only stdout. Human mode still prints the text report; `--json` prints the JSON instead.
- Do not mock the filesystem package. Use `envs/echo_env` and `tests/fixtures/validation/*` as committed.

After a JSON report is written, check it:

```bash
uv run python "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/check-report.py" \
  "$EVIDENCE_ROOT/<feature-id>/report.json"
```

## Cleanup

```bash
bash "$REPO_ROOT/.cursor/skills/verify-openenv-validate/bin/cleanup.sh"
```

Removes only `$SCRATCH_ROOT` (`/tmp/openenv-validate-verify-$RUN_ID`). Does not kill by process name. Does not delete `$EVIDENCE_ROOT`. After cleanup, `ls "$EVIDENCE_ROOT/<feature-id>"` must still show the proof files.

## Helpers

All scripts are executable. Invoke them as shown; do not invent flags.

| Script | Purpose |
| --- | --- |
| `bin/doctor.sh` | Read-only readiness. No args. Exit 0/1. |
| `bin/validate-run.sh <feature-id> -- <openenv-args…>` | One isolated `uv run openenv` capture. |
| `bin/check-report.py <report.json>` | `ValidationReport` parse; prints verdict and `static.manifest` status. |
| `bin/cleanup.sh` | Delete scratch dir only. |

`validate-run.sh` requires `REPO_ROOT`, `EVIDENCE_ROOT` from Launch. Optional `EXPECTED_EXIT` (default: do not assert). If `EXPECTED_EXIT` is set and the process exits otherwise, the helper exits 1 after still writing the capture files.
