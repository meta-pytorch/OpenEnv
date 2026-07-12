# CLI

The `openenv` CLI provides a set of commands for building, validating, and pushing environments to Hugging Face Spaces or a custom Docker registry. For an end-to-end tutorial on building environments with OpenEnv, see the [building an environment](../getting_started/environment-builder.md) guide.

## `openenv init`

[[autodoc]] openenv.cli.commands.init.init

## `openenv import`

Import a supported third-party source environment into a generated OpenEnv
wrapper package. The command detects the source format from the directory
contents, so ORS/OpenReward and Prime Intellect Verifiers sources do not
require `--type` in the common case.

The generated wrapper vendors the source tree into the package and includes
vendored files as package data, so non-secret fixture/data files are available to
the environment server at runtime. The importer carries portable dependencies
from source `pyproject.toml` and `requirements.txt` files into the generated
environment, skips VCS/cache/build directories and common secret file patterns
such as `.env`, `secrets.yaml`, and private key files, and excludes compiled
binary artifacts; review the generated `vendor/` directory before publishing a
wrapper.

```bash
openenv import path/to/source --name my_env --output-dir ./envs
openenv import path/to/source --name my_env --output-dir ./envs --env-class MyEnv
```

```{eval-rst}
.. automodule:: openenv.cli.commands.import_env
   :members:
   :undoc-members:
   :show-inheritance:
```

## `openenv build`

[[autodoc]] openenv.cli.commands.build.build

## `openenv validate`

Use an explicit profile to produce the RFC 008 validation plan and shared
report schema:

```bash
openenv validate path/to/env --profile static
openenv validate path/to/env --profile runtime --json
openenv validate --url http://127.0.0.1:8000 --profile runtime --json
openenv validate path/to/env --profile full --output validation.json
openenv validate path/to/env --profile publish --remote --output validation.json
```

`static` checks source and packaging, `runtime` adds a launched or connected
server, and `full` records every policy criterion while marking unavailable
remote capabilities as skipped. `publish` runs the runtime check set as a
strict author gate: every blocking criterion must pass, so blocking skips are
reported as incomplete and exit non-zero. Reports include typed diagnostics,
repository-relative locations, display-only remediation, and safe evidence
with `--verbose`.

`--remote` uploads the environment revision and the initiating validator source
to a new dedicated Hugging Face Sandbox. The HF token authorizes Sandbox
creation but is not forwarded into the workload. The returned report uses the
same schema and remains unofficial; it never claims Hub certification.
Automatic local launch executes the checkout as the current user and is only
for trusted development source. Use `--url` for a server you already isolated,
or `--remote` to isolate source execution.

This command intentionally targets the served OpenEnv spec. Shared reports
record spec, adapter, and execution-model provenance, but `openenv validate`
does not auto-dispatch external task-package formats. A separate spec-selected
task workflow is tracked in [issue #898](https://github.com/huggingface/OpenEnv/issues/898).

[[autodoc]] openenv.cli.commands.validate.validate

## `openenv push`

Hub pushes automatically require a passing remote `publish` report. The CLI
adds a portable copy to `.openenv/validation-report.json` in the same uploaded
revision. This is author evidence for debugging and automation, not the richer
independent certification report that the planned Hub validation service will
produce.

[[autodoc]] openenv.cli.commands.push.push

## `openenv serve`

Local serving is not implemented in the CLI yet. This command exits non-zero
and prints alternative ways to run an environment server.

[[autodoc]] openenv.cli.commands.serve.serve

## `openenv fork`

[[autodoc]] openenv.cli.commands.fork.fork

## `openenv skills`

Installs an `openenv-cli` skill into your AI assistant's skills directory so
it knows the `openenv` CLI is available and what each command does. Supports
Claude Code, Cursor, Codex, and OpenCode.

**Install for a single assistant (project-local):**

```bash
openenv skills add --claude    # → .claude/skills/openenv-cli/
openenv skills add --cursor    # → .cursor/skills/openenv-cli/
openenv skills add --codex     # → .codex/skills/openenv-cli/
openenv skills add --opencode  # → .opencode/skills/openenv-cli/
```

Multiple flags can be combined — `openenv skills add --claude --cursor` installs
for both at once. The skill file is written to a central location
(`.agents/skills/openenv-cli/`) and each agent directory gets a symlink, so
there is only one copy to update.

**Install globally (user-level, across all projects):**

```bash
openenv skills add --claude --global  # → ~/.claude/skills/openenv-cli/
```

**Overwrite an existing installation** (e.g. after upgrading `openenv`):

```bash
openenv skills add --claude --force
```

**Preview the skill content without installing:**

```bash
openenv skills preview
```

**Install to a custom path** (for non-standard agent setups):

```bash
openenv skills add --dest /path/to/my-agent/skills/
```

[[autodoc]] openenv.cli.commands.skills.skills_add

[[autodoc]] openenv.cli.commands.skills.skills_preview

# API Reference

## Entry point

[[autodoc]] openenv.cli.__main__.main

## CLI helpers

[[autodoc]] openenv.cli._cli_utils.validate_env_structure

## Validation utilities

[[autodoc]] openenv.cli._validation.validate_running_environment

[[autodoc]] openenv.cli._validation.validate_multi_mode_deployment

[[autodoc]] openenv.cli._validation.get_deployment_modes

[[autodoc]] openenv.cli._validation.format_validation_report

[[autodoc]] openenv.cli._validation.build_local_validation_json_report
