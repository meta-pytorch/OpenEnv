# `openenv validate` verification map

This directory is the maintained source for verifying the user-facing behavior of `openenv validate`. Read this index before driving the CLI, then use the matching feature file as the recipe.

## Baseline preconditions

- Work from `$REPO_ROOT` (OpenEnv checkout). `uv` is on `PATH` (`~/.local/bin`).
- `bash .cursor/skills/verify-openenv-validate/bin/doctor.sh` exits 0.
- `EVIDENCE_ROOT` and `SCRATCH_ROOT` are set from the skill Launch section.
- Do not mutate `envs/echo_env/` or `tests/fixtures/validation/`.
- Do not start an environment server unless a feature file names `--url`.
- Treat every command as literal. Keep package paths and flags unchanged.

## Driving conventions

- Start every recipe from the baseline unless its preconditions say otherwise.
- Drive through `bin/validate-run.sh` so stdout, stderr, and the exit code are captured together.
- Prefer `uv run openenv validate …` (user path). The module form `python -m openenv.cli validate` is the same CLI; use it only when a recipe says so.
- Local static recipes always pass `--level static --skip-build`.
- Restore nothing after a drive: these commands are read-only on package trees. Scratch reports under `$SCRATCH_ROOT` are removed in cleanup. Proof under `$EVIDENCE_ROOT` stays.

## Proof and skip reporting

- CLI proof includes the command, stdout, stderr, and exit code.
- JSON proof is a file that `bin/check-report.py` accepts.
- Record the feature ID with every artifact directory name.
- Report an unreachable path with the attempted command and the unmet precondition.
- Do not report a skipped entry point as verified through a different path (human vs `--json` vs `--output` are distinct when the feature lists them).

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features`
2. `How to get to it (user POV)`
3. `Driving it with validate-run`
4. `Gotchas`

## Features

- [Echo env static validate](./echo-env-static-validate.md) is the slice-1 happy path: `envs/echo_env` exits 0 with a schema-valid report and `static.manifest` PASS.
- [Broken manifest](./broken-manifest.md) is a graded FAIL: invalid `validation:` exits 1 and names `static.manifest`.
- [Ambiguous package](./ambiguous-package.md) is a signature refusal: two well-known files exit 2 on stderr, no report.
- [JSON report output](./json-report-output.md) is the machine-readable path: `--json` and `--output` emit `ValidationReport`.
- [Unrecognized package](./unrecognized-package.md) is a signature refusal: no well-known file exits 2 on stderr.
