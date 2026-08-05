"""Local subclasses that fix agent INSTALL failures, registered via `AgentConfig.import_path`.

Same escape hatch as `pi_agent.py`, different reason. These agents' seams are fine; they simply never
get far enough to use them, because installing the CLI into the DataAgent sandbox fails. Both are
upstream bugs in Harbor's wrappers rather than anything about the intercept, and both are fixed here
without touching Harbor.

Each override is deliberately minimal: call Harbor's own `install()` and change only the one thing
that is wrong, so we inherit every future upstream fix instead of forking the install logic.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.cline.cline import ClineCli, ExecInput
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.agents.installed.hermes import Hermes
from harbor.agents.installed.kimi_cli import KimiCli
from harbor.agents.installed.openclaw import OpenClaw
from harbor.agents.installed.openhands import OpenHands
from harbor.agents.installed.pi import Pi
from harbor.agents.installed.swe_agent import SweAgent
from harbor.environments.base import BaseEnvironment


class InterceptGeminiCli(GeminiCli):
    """gemini-cli, with `bash` guaranteed before nvm runs.

    `nvm_node_install_snippet()` pipes the installer into **bash**:

        curl -o- .../install.sh | env -u NODE_VERSION bash

    In an image without bash that pipe fails, nvm never lands, and the snippet's own guard reports:

        Error: NVM failed to load

    which reads like an nvm problem rather than a missing shell. Harbor 0.20.0's own
    `GeminiCli.install` apt-installs `curl` and nothing else (gemini_cli.py:110-115), so `bash` is
    still the gap this closes.

    Installed directly with `exec_as_root` rather than through a dependency helper: the
    `ensure_system_dependencies(environment, (...))` this used to call does not exist in Harbor
    0.20.0, and calling it failed the install outright with

        'InterceptGeminiCli' object has no attribute 'ensure_system_dependencies'

    — which surfaced as `Agent install failed` and zero model calls, i.e. a harness that could not run
    at all. Mirrors Harbor's own apt invocation in the same file so the two cannot drift again.
    """

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y curl bash",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await super().install(environment)


class InterceptOpenHands(OpenHands):
    """OpenHands pinned to the last V0 release, which still has `openhands.core.main`.

    Harbor's wrapper installs `openhands-ai` unpinned and then verifies with:

        /opt/openhands-venv/bin/python -m openhands.core.main --version

    OpenHands V1 restructured the package: the agent core moved out into `openhands-sdk` /
    `openhands-agent-server`, so `openhands.core` no longer exists and install fails with

        ModuleNotFoundError: No module named 'openhands.core'

    V0 is scheduled for removal on 2026-04-01, so this pin is a stopgap: once Harbor's wrapper is
    updated for the V1 entry point, drop the pin and this subclass. 0.49.0 is the newest 0.x on PyPI.
    """

    DEFAULT_V0_VERSION = "0.49.0"
    # Harbor defaults to `uv python install 3.13`, but 0.49.0 shipped in July 2025 and does not
    # resolve there, so the version pin alone is not enough.
    DEFAULT_PYTHON = "3.12"

    # Transitive dependencies openhands-ai 0.49.0 imports but does not declare. Installing 0.49.0
    # succeeds, and then the import check dies:
    #   File ".../openhands/events/event_store_abc.py", line 5, in <module>
    #       from deprecated import deprecated
    #   ModuleNotFoundError: No module named 'deprecated'
    # Pinning an old release means living with whatever its metadata got wrong at the time.
    MISSING_DEPS = ("Deprecated",)
    VENV = "/opt/openhands-venv"

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("version", self.DEFAULT_V0_VERSION)
        kwargs.setdefault("python_version", self.DEFAULT_PYTHON)
        super().__init__(*args, **kwargs)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install via Harbor, and repair the missing deps if its verify step trips over them.

        Harbor's install ends with `python -m openhands.core.main --version`, so an undeclared
        dependency surfaces as a failed install rather than a failed run. We cannot pre-empt it
        without forking Harbor's install command, so instead: let it run, and if it fails, add the
        known-missing packages to the venv it already built and re-run the same verification. If that
        passes, the install is genuinely fine.
        """
        try:
            await super().install(environment)
            return
        except Exception as exc:  # noqa: BLE001 - remediate, then re-verify honestly
            self.logger.warning(
                "openhands install failed (%s); attempting dependency repair",
                str(exc)[:160],
            )

        packages = " ".join(self.MISSING_DEPS)
        await self.exec_as_agent(
            environment,
            command=(
                f"set -euo pipefail; {self.VENV}/bin/python -m ensurepip --upgrade || true; "
                f"{self.VENV}/bin/python -m pip install {packages}"
            ),
        )
        await self._install_poetry_shim(environment)
        # Same check Harbor uses. If this still fails it raises, and the failure is real.
        await self.exec_as_agent(
            environment,
            command=f"{self.VENV}/bin/python -m openhands.core.main --version",
        )

    async def _install_poetry_shim(self, environment: BaseEnvironment) -> None:
        """Make `poetry run python ...` work in a venv that poetry never created.

        OpenHands' LocalRuntime starts its action-execution server with:

            ['poetry', 'run', 'python', '-u', '-m', 'openhands.runtime.action_execution_server', ...]

        which assumes a poetry-managed source checkout. Installed from PyPI into a uv venv there is no
        pyproject.toml anywhere above site-packages, so poetry refuses:

            server: Poetry could not find a pyproject.toml file in
                    /opt/openhands-venv/lib/python3.12/site-packages or its parents
            server process exited

        The agent then waits for a server that will never come up, and tenacity converts that into
        `RetryError[<Future ... raised RuntimeError>]` with the actual cause nowhere in the traceback.

        Rather than fabricate a pyproject.toml (which makes poetry resolve and possibly reinstall a
        dependency tree), shim the one invocation OpenHands makes: drop the `run` verb and exec the
        venv's own interpreter. Everything the server needs is already installed there.
        """
        shim = (
            "#!/bin/sh\n"
            '[ "$1" = "run" ] && shift\n'
            f'[ "$1" = "python" ] && {{ shift; exec {self.VENV}/bin/python "$@"; }}\n'
            f'exec {self.VENV}/bin/"$@"\n'
        )
        # Installed as ROOT into /usr/local/bin, and over any existing poetry.
        #
        # A first attempt wrote it to ~/.local/bin and changed nothing: the error was
        # "Poetry could not find a pyproject.toml", not "poetry: command not found", so a real poetry
        # is already on PATH ahead of ~/.local/bin. Shadowing it is the only way the shim is reached.
        # /usr/local/bin precedes ~/.local/bin on every image we run.
        for directory in ("/usr/local/bin", "/usr/bin"):
            await self.exec_as_root(
                environment,
                command=(
                    f"mkdir -p {directory} && cat > {directory}/poetry <<'SHIM'\n{shim}SHIM\n"
                    f"chmod 0755 {directory}/poetry"
                ),
            )


class InterceptSweAgent(SweAgent):
    """swe-agent, given a git repo to work in so Harbor's working code path is taken.

    Harbor builds the repo argument as:

        "$(if [ -d /testbed ]; then echo '--env.repo.type=preexisting --env.repo.repo_name=/testbed'; "
        "else echo '--env.repo.path=$(pwd)'; fi)"

    The else-branch is broken: `$(pwd)` sits inside SINGLE quotes, so it is never expanded and the
    literal string is passed through. swe-agent then resolves it relative to its cwd and dies:

        git.exc.NoSuchPathError: /workdir/$(pwd)

    Underneath that is a second problem: swe-agent is a SWE-bench agent and requires a git repository,
    while DataAgent tasks are a CSV and a question.

    Both are solved by satisfying the `[ -d /testbed ]` test that Harbor already checks. We `git init`
    the task's own /workdir and expose it as /testbed, so Harbor takes its preexisting-repo branch
    (which has no quoting bug) and the agent still works where the task data and /workdir/answer.txt
    live. Nothing in Harbor changes.
    """

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self.exec_as_root(
            environment,
            command=(
                "set -eu; mkdir -p /workdir; cd /workdir; "
                # A repo with no commit still fails some checks, so make one.
                "git rev-parse --git-dir >/dev/null 2>&1 || { "
                "  git init -q .; "
                "  git config user.email harbor@example.com; git config user.name harbor; "
                "  touch .harbor-keep; git add -A; git commit -qm 'harbor: initial' || true; }; "
                "[ -e /testbed ] || ln -s /workdir /testbed"
            ),
        )


# Runs INSIDE the sandbox. Truncates `openclaw.txt` after the last line that is exactly `}`, i.e.
# the closing brace of openclaw's pretty-printed `--json` envelope. Deliberately not a JSON parser:
# re-implementing Harbor's scan here is exactly what we are trying to avoid, so this only removes
# the trailing lines and then lets Harbor's own parser do the parsing.
_OPENCLAW_TRIM_TRAILING_LOG = """
from pathlib import Path

p = Path("/logs/agent/openclaw.txt")
if p.is_file():
    lines = p.read_text(encoding="utf-8", errors="replace").rstrip().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == "}":
            if i < len(lines) - 1:
                p.write_text("\\n".join(lines[: i + 1]) + "\\n", encoding="utf-8")
            break
"""


class InterceptOpenClaw(OpenClaw):
    """openclaw, with its config actually present inside the sandbox.

    Harbor writes the merged config to the HOST trial dir and then copies it from a CONTAINER path:

        upload_path = self.logs_dir / "openclaw.upload.json"        # host
        "mkdir -p ~/.openclaw && cp /logs/agent/openclaw.upload.json ~/.openclaw/openclaw.json"

    with the comment "trial mounts logs here as /logs/agent". That holds for a bind-mounted docker
    runtime. **E2B has no bind mounts**, so the file exists on the host and nowhere in the sandbox:

        cp: cannot stat '/logs/agent/openclaw.upload.json'

    Supplying `openclaw_config` does not help, because the problem is not that the config is empty.

    Fix: build the same config Harbor would and upload it to the container path during setup, so the
    copy in `run()` finds it. Harbor's own `_build_full_openclaw_config` is reused, so the content
    stays whatever Harbor intended, merges included.
    """

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)

        payload = json.dumps(self._build_full_openclaw_config(), indent=2) + "\n"
        local = Path(self.logs_dir) / self._UPLOAD_CONFIG_FILENAME
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(payload, encoding="utf-8")

        target = f"{self._CONTAINER_LOGS_AGENT}/{self._UPLOAD_CONFIG_FILENAME}"
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {self._CONTAINER_LOGS_AGENT} && "
            f"chmod 777 {self._CONTAINER_LOGS_AGENT}",
        )
        await environment.upload_file(local, target)

    async def _copy_openclaw_session_file_to_agent_logs(
        self, environment: BaseEnvironment, env: dict[str, str]
    ) -> None:
        """Strip openclaw's trailing stderr line so Harbor can parse its own capture file.

        UPSTREAM HARBOR BUG. Delete this override once Harbor's parser tolerates trailing text.

        Harbor runs openclaw as (openclaw.py:947-953):

            openclaw agent --local --json ... 2>&1 </dev/null | stdbuf -oL tee /logs/agent/openclaw.txt

        openclaw writes clean JSON to STDOUT; that `2>&1` merges its STDERR into the same file. After
        the envelope is flushed, openclaw logs one info-level line to stderr:

            [agents/agent-command] [agent] run <uuid> ended with stopReason=stop

        Harbor then parses that file with a rule requiring the JSON object to consume the entire
        remaining suffix (`_openclaw_decode_last_json_dict_suffix`, and the identical loop inside
        `_openclaw_container_copy_session_transcript`). One trailing line defeats both, so the
        container-side copy hits `sys.exit(0)` and `populate_context_post_run` returns at
        `if not envelope: return` -- no `openclaw.session.jsonl` and no `trajectory.json` at all,
        while the session file sits on disk the whole time. Verified by stream: the line appears in
        stderr and never in stdout.

        Probably unnoticed upstream because the line is suppressed for `stopReason == "end_turn"`
        (dist/agent-command:454). Anthropic reports `end_turn`; every OpenAI-compatible provider
        reports `stop`, so for us it is always printed.

        The fix is deliberately NOT a local re-implementation of Harbor's parser. Removing the
        trailing lines leaves Harbor's own scan -- container-side and host-side -- to run unmodified,
        so any upstream improvement to it is still inherited.
        """
        try:
            await self.exec_as_agent(
                environment,
                command="python3 -c " + shlex.quote(_OPENCLAW_TRIM_TRAILING_LOG),
                env=env,
            )
        except Exception as exc:  # noqa: BLE001 - a missing trace must not fail a good rollout
            # Loud, unlike Harbor's silent `sys.exit(0)`: losing the trajectory is the whole bug.
            self.logger.warning(
                "could not trim openclaw.txt (%s); ATIF trajectory will likely be missing",
                str(exc)[:160],
            )
        await super()._copy_openclaw_session_file_to_agent_logs(environment, env)


class InterceptCline(ClineCli):
    """cline-cli, given a base URL through the only channel it has: its settings store.

    Harbor forwards exactly `{PROVIDER, API_KEY, MODELID}` and runs

        cline -P <provider> -k $API_KEY -m $MODELID --json --yolo

    There is no base-URL flag and no base-URL env var, so cline resolves the provider's REAL endpoint
    and dies with the session id as a bearer token:

        Incorrect API key provided: s55f2f5a… You can find your API key at
        https://platform.openai.com/account/api-keys

    Cline's OpenAI-Compatible provider takes Base URL + key + model id from its settings store
    (`~/.cline/data/globalState.json`), not from the CLI. Harbor writes that file itself at the start
    of `create_run_agent_commands`, so anything written earlier is overwritten. Instead we let
    Harbor's command run and INSERT a merge step between it and the agent invocation, which keeps
    Harbor's own keys (`welcomeViewCompleted`, `isNewUser`) intact.
    """

    def __init__(
        self, *args: Any, intercept_config: dict[str, str] | None = None, **kwargs: Any
    ):
        self._intercept_config = intercept_config or {}
        super().__init__(*args, **kwargs)

    def create_run_agent_commands(self, instruction: str):
        commands = list(super().create_run_agent_commands(instruction))
        base_url = self._intercept_config.get("base_url")
        api_key = self._intercept_config.get("api_key")
        model = self._intercept_config.get("model")
        if not (base_url and api_key and model) or not commands:
            return commands

        # Merge rather than replace: Harbor's globalState keys must survive.
        settings = {
            "openAiBaseUrl": f"{base_url}/v1",
            "openAiApiKey": api_key,
            "openAiModelId": model,
            "apiProvider": "openai",
        }
        merge = (
            "python3 - <<'__HARBOR_CLINE_SETTINGS__'\n"
            "import json, pathlib\n"
            "p = pathlib.Path.home() / '.cline' / 'data' / 'globalState.json'\n"
            "p.parent.mkdir(parents=True, exist_ok=True)\n"
            "try:\n"
            "    cfg = json.loads(p.read_text())\n"
            "except Exception:\n"
            "    cfg = {}\n"
            f"cfg.update({json.dumps(settings)})\n"
            "p.write_text(json.dumps(cfg))\n"
            "__HARBOR_CLINE_SETTINGS__"
        )
        commands.insert(1, ExecInput(command=merge))
        return commands


class InterceptHermes(Hermes):
    """hermes, pointed at our endpoint via its `custom_providers` config, compression off.

    Harbor generates hermes' config.yaml with `provider: "auto"`, so hermes routes itself and never
    honours OPENAI_BASE_URL. It authenticates somewhere else entirely and the intercept records zero
    calls:

        HTTP 401: Missing Authentication header

    Pinning `provider: "openai"` does NOT work -- hermes rejects it outright
    (`Unknown provider 'openai'`), because its table maps the openai prefix to a `None` CLI flag,
    i.e. "use the full provider/model name and let auto-routing resolve it".

    The supported route is a **custom provider** entry
    (https://hermes-agent.nousresearch.com/docs/integrations/providers):

        custom_providers:
          - name: intercept
            base_url: http://host:port/v1
            api_key: ...

    Compression is also disabled. Harbor enables it (`threshold: 0.85`), and compaction rewrites the
    conversation, which breaks token-prefix stitching: the rollout would fragment into per-turn rows
    exactly as terminus-2 does.

    `_build_config_yaml` is a staticmethod upstream but is invoked as `self._build_config_yaml(...)`,
    so overriding it as an instance method is safe and gives access to the injected endpoint.
    """

    PROVIDER = "intercept"

    def __init__(
        self, *args: Any, intercept_config: dict[str, str] | None = None, **kwargs: Any
    ):
        self._intercept_config = intercept_config or {}
        super().__init__(*args, **kwargs)

    def _build_config_yaml(self, model: str) -> str:  # type: ignore[override]
        import yaml

        cfg = yaml.safe_load(Hermes._build_config_yaml(model)) or {}

        base_url = self._intercept_config.get("base_url")
        api_key = self._intercept_config.get("api_key")
        if base_url:
            entry = {
                "name": self.PROVIDER,
                "base_url": f"{base_url}/v1",
                "model": model,
            }
            if api_key:
                entry["api_key"] = api_key
            cfg["custom_providers"] = [entry]
            cfg["provider"] = self.PROVIDER

        # Compaction is incompatible with prefix stitching; see docs/HARNESS_NOTES.md.
        cfg.setdefault("compression", {})["enabled"] = False
        return yaml.safe_dump(cfg, sort_keys=False)

    async def run(self, instruction, environment, context) -> None:  # type: ignore[override]
        """Re-export the session without `--source cli`, which matches nothing.

        UPSTREAM HARBOR BUG. Delete this override once Harbor drops the flag (hermes.py:446).

        Harbor ends a hermes run with:

            hermes sessions export /logs/agent/hermes-session.jsonl --source cli 2>/dev/null || true

        That command does not fail -- it succeeds and exports NOTHING, leaving a 0-byte file, so
        `populate_context_post_run` reads an empty session and writes no `trajectory.json`.

        The flag is not the obvious suspect: the session's `source` really is `cli`
        (`hermes sessions list` prints `Src=cli`). The actual mechanism is that in hermes-agent ANY
        filter switches export off `db.export_all()` and onto `db.list_prune_candidates()`
        (hermes_cli/sessions_cmd.py:329-379), whose WHERE clause opens with

            clauses = ["s.ended_at IS NOT NULL"]      # hermes_state.py:7852

        documented as "Only ended sessions are ever candidates ... so a live session is never
        selected". And `hermes chat -q ... -Q` never finalizes its session, so `ended_at` stays NULL:

            sqlite> SELECT id, source, ended_at FROM sessions;
            20260803_074252_1a49f2|cli|

        That second half is a hermes-agent behaviour we do not control (reproduced with a plain
        `hermes chat -q` run, no custom provider involved). What Harbor controls is passing a filter
        that provably selects zero rows for exactly the sessions it creates. Without the flag the
        same export returns the session, and Harbor's own unmodified converter turns it into a
        valid ATIF trajectory.

        Re-running the export afterwards rather than editing Harbor's command is what keeps this an
        override instead of a fork: Harbor's version runs first in its own `finally`, writes the
        0-byte file, and this overwrites it before the logs are downloaded.
        """
        await super().run(instruction, environment, context)
        try:
            await self.exec_as_agent(
                environment,
                command=(
                    'export PATH="$HOME/.local/bin:$PATH" && '
                    "hermes sessions export /logs/agent/hermes-session.jsonl"
                ),
                env={"HERMES_HOME": "/tmp/hermes"},
                timeout_sec=60,
            )
        except Exception as exc:  # noqa: BLE001 - a missing trace must not fail a good rollout
            # Warned rather than swallowed: Harbor's `2>/dev/null || true` is what hid this for so
            # long, and a silent second copy of that mistake would be worse than the first.
            self.logger.warning(
                "unfiltered hermes session export failed (%s); ATIF trajectory will be missing",
                str(exc)[:160],
            )


class InterceptKimi(KimiCli):
    """kimi-cli, surviving the stream reset its own teardown causes.

    Harbor runs kimi as (kimi_cli.py:379-394):

        (echo $PROMPT; sleep 86400) | kimi --wire --yolo --afk ... | (
            while IFS= read -r line; do ... case "$line" in *'"id":"1"'*) break ;; esac; done
            ...; kill 0)

    `sleep 86400` holds stdin open for a day, and `kill 0` tears down the whole process group once
    the terminating wire event arrives. Harbor already expects part of the fallout and swallows
    `NonZeroAgentExitCodeError` for "exit 143" (SIGTERM). What it does not expect is that killing the
    group also kills the E2B exec stream mid-flight, so the HTTP/2 connection on the HOST side dies:

        httpcore.RemoteProtocolError: <StreamReset stream_id:67, error_code:2, remote_reset:True>
        (raised in .venv312/site-packages/httpcore/_async/http2.py, i.e. OUR process, not the sandbox)

    That propagates out of `run`, so Harbor abandons the trial and never runs the verifier. Every one
    of 11 kimi trials died this way, each AFTER completing real work (one had 37 captured turns), and
    no other harness has ever produced this error on the same E2B backend, which is what identifies
    it as kimi's teardown rather than transport flakiness.

    By the time it fires, kimi has already written its wire output to /logs/agent/, so the trajectory
    and answer are on disk and the trial can be graded normally. Swallowing it here is the same
    judgement Harbor already made for exit 143, applied to the other half of the same teardown.

    The same teardown has more than one spelling, which is what `_TEARDOWN_ERRORS` is for. Against
    Harbor 0.20.0 every kimi rollout instead raised

        httpx.ConnectError: Error reading content

    so the original `RemoteProtocolError`-only guard no longer matched and all 15 cells of a
    compatibility matrix failed — each one AFTER capturing real work (up to 12 turns and 1320
    trainable tokens, `atif=match` throughout). One transport layer's way of saying "the stream you
    were reading went away" is not stable across versions, so the guard lists the ways rather than
    assuming one.

    Still deliberately narrow, and the safety net is downstream rather than here: if the sandbox had
    genuinely been unreachable, the agent would have made no model calls, and `check_rollout`'s
    `no_turns` FATAL fails the rollout anyway. So swallowing a transport error cannot promote a
    never-ran rollout to a graded one. Anything outside this list still raises.
    """

    # (exception class name, substring that identifies it as the exec stream dying)
    _TEARDOWN_ERRORS = (
        ("RemoteProtocolError", "StreamReset"),
        ("ConnectError", "Error reading content"),
    )

    async def run(self, instruction, environment, context) -> None:  # type: ignore[override]
        try:
            await super().run(instruction, environment, context)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is the known teardown
            name, text = type(exc).__name__, str(exc)
            if not any(
                name == cls and marker in text for cls, marker in self._TEARDOWN_ERRORS
            ):
                raise
            # Expected: `kill 0` took the exec stream down with the process group.


# ------------------------------------------------------------------------------------------------
# pi
# ------------------------------------------------------------------------------------------------
# pi, taught to talk to our intercept. Registered via `AgentConfig.import_path`, no Harbor patch.
#
# THE PROBLEM. Harbor's `pi` wrapper forwards a fixed list of API-key variables and nothing else
# (installed/pi.py:100-137). It has no base-URL handling at all, and unlike opencode there is no
# `_build_register_config_command` hook to write a provider config. Pointed at our intercept, pi ignores
# `OPENAI_BASE_URL`, calls api.openai.com with our session id as the key, and dies:
#
#     OpenAI API error (401): Incorrect API key provided: sc6f2e5a…
#     You can find your API key at https://platform.openai.com/account/api-keys
#
# THE FIX. pi reads custom providers from `~/.pi/agent/models.json`
# (https://pi.dev/docs/latest/custom-provider). Harbor gives us no hook to write it, but it does let an
# agent be supplied by `import_path`, so we subclass `Pi`, write the file in `setup()`, and register
# the subclass. Harbor is untouched.
#
#     AgentConfig(import_path="harnesses.pi_agent:InterceptPi", ...)
#
# This is the general escape hatch for any harness whose config Harbor does not know how to write:
# subclass locally, override `setup()`, register by import path.
#
# TWO DETAILS THAT MATTER.
#
# `api` must be `openai-completions`. Left to its own devices pi picks `openai-responses` for a provider
# named `openai` (we watched it do exactly that: `"api":"openai-responses"` in its session log). The
# intercept handles both, but chat-completions is the dialect with the least translation and by far the
# most mileage on it.
#
# The provider is NOT named `openai`. A distinct name keeps pi off its built-in OpenAI defaults, the
# same reason opencode's provider is called `intercepted`.
PROVIDER = "intercept"
MODELS_JSON = "~/.pi/agent/models.json"


def build_models_json(base_url: str, api_key: str, model: str) -> str:
    return json.dumps(
        {
            "providers": {
                PROVIDER: {
                    "baseUrl": f"{base_url}/v1",
                    "api": "openai-completions",
                    "apiKey": api_key,
                    "models": [{"id": model}],
                }
            }
        },
        indent=2,
    )


class InterceptPi(Pi):
    """`Pi` that writes a custom-provider config into the sandbox before running.

    Config arrives through `AgentConfig.kwargs` as `intercept_config`, mirroring how opencode
    receives `opencode_config`, so the seam table stays uniform across harnesses.
    """

    def __init__(
        self, *args: Any, intercept_config: dict[str, str] | None = None, **kwargs: Any
    ):
        self._intercept_config = intercept_config or {}
        super().__init__(*args, **kwargs)

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)

        base_url = self._intercept_config.get("base_url")
        api_key = self._intercept_config.get("api_key")
        model = self._intercept_config.get("model")
        if not (base_url and api_key and model):
            # Refuse quietly rather than run against api.openai.com with a session id as the key,
            # which is what happens by default and costs a sandbox to discover.
            raise ValueError(
                "InterceptPi requires intercept_config with base_url, api_key, model"
            )

        payload = shlex.quote(build_models_json(base_url, api_key, model))
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p ~/.pi/agent && printf '%s' {payload} > {MODELS_JSON}",
        )
