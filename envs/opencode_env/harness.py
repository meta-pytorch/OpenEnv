# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode session factory + session implementation.

Implements the :class:`ResourceSessionFactory` / :class:`ResourceSession`
contracts from ``openenv.core.harness`` (PR #471). The session wraps one E2B
(or other backend) sandbox running the OpenCode CLI agent.

Mode A (black_box) is implemented here. Mode B (transparent_proxy) layers on
top in :mod:`opencode_env.interception` without changing this file's contract.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)

from .config import OpenCodeConfig
from .interception import read_trace as _read_proxy_trace
from .opencode_runtime import (
    agent_log_path,
    build_env_vars,
    build_install_cmd,
    build_opencode_json,
    build_run_cmd,
    instruction_path,
    opencode_config_path,
    system_prompt_path,
)
from .sandbox.base import BgJob, SandboxBackend, SandboxHandle
from .task import OpenCodeTask


# Inside-sandbox paths for the interception proxy (Mode B).
_PROXY_PORT = 7000
_PROXY_TRACE_PATH = "/home/user/logs/agent/proxy_trace.jsonl"
_PROXY_LOG_PATH = "/home/user/logs/agent/proxy.log"


Verifier = Callable[[SandboxHandle, OpenCodeTask], VerifyResult]


class OpenCodeSession(ResourceSession):
    """One live OpenCode rollout inside a sandbox.

    The session is created already-running: :meth:`OpenCodeSessionFactory.create`
    calls :meth:`start_agent` before returning. Typical usage:

    .. code-block:: python

        session = factory.create(task)
        session.wait_for_completion()
        result = session.verify([])
        session.close()
    """

    def __init__(
        self,
        *,
        sandbox: SandboxHandle,
        config: OpenCodeConfig,
        task: OpenCodeTask,
        verifier: Verifier | None = None,
        base_url_override: str | None = None,
        proxy_trace_path: str | None = None,
        proxy_bg_job: BgJob | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.config = config
        self.task = task
        self._verifier = verifier
        self._base_url_override = base_url_override
        self._bg_job: BgJob | None = None
        self._proxy_trace_path = proxy_trace_path
        self._proxy_bg_job = proxy_bg_job

    # ------------------------------------------------------------------
    # ResourceSession contract (PR #471)
    # ------------------------------------------------------------------
    def initial_messages(self) -> list[Message]:
        return [{"role": "user", "content": self.task.instruction}]

    def list_tools(self) -> list[Tool]:
        # OpenCode discovers its own tools from its LLM provider; none are
        # exposed to the harness directly.
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=(
                "OpenCodeSession does not expose external tool calls; the "
                "CLI agent owns its own tool loop."
            )
        )

    def verify(
        self,
        transcript: list[Message],
        final_state: Any | None = None,
    ) -> VerifyResult:
        if self._verifier is None:
            return VerifyResult(env_reward=None, done=True)
        return self._verifier(self.sandbox, self.task)

    def close(self) -> None:
        if self._bg_job is not None:
            try:
                self._bg_job.kill()
            except Exception:
                pass
            self._bg_job = None
        if self._proxy_bg_job is not None:
            try:
                self._proxy_bg_job.kill()
            except Exception:
                pass
            self._proxy_bg_job = None
        self.sandbox.kill()

    # ------------------------------------------------------------------
    # OpenCode-specific session API
    # ------------------------------------------------------------------
    def start_agent(self) -> None:
        """Launch ``opencode run`` as a background process.

        Called by :meth:`OpenCodeSessionFactory.create` after all sandbox
        staging is done. Idempotent: a second call is a no-op.
        """
        if self._bg_job is not None:
            return
        cmd = build_run_cmd(self.config)
        envs = build_env_vars(self.config, base_url_override=self._base_url_override)
        self._bg_job = self.sandbox.start_bg(cmd, envs=envs)

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Block until OpenCode exits; return its exit code."""
        if self._bg_job is None:
            raise RuntimeError("Agent not started; call start_agent() first.")
        timeout = timeout_s if timeout_s is not None else self.config.agent_timeout_s
        return self._bg_job.wait(timeout=timeout)

    def fetch_trace(self) -> str:
        """Return the raw ``opencode run`` log (JSON-lines when ``run_format=json``)."""
        return self.sandbox.read_text(agent_log_path(self.config))

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Return per-turn proxy-captured records (Mode B only).

        Each entry has ``request``, ``response``, ``completion_tokens``,
        ``completion_token_ids``, ``per_token_logps``, ``finish_reason``, and
        ``latency_s``. Returns ``[]`` in Mode A (no proxy).
        """
        if self._proxy_trace_path is None:
            return []
        try:
            content = self.sandbox.read_text(self._proxy_trace_path)
        except Exception:
            return []
        records: list[dict[str, Any]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            import json as _json
            records.append(_json.loads(line))
        return records


class OpenCodeSessionFactory(ResourceSessionFactory):
    """Produce isolated per-rollout :class:`OpenCodeSession` instances.

    The factory owns sandbox provisioning, OpenCode install, config injection,
    and optional task-specific setup. Each call to :meth:`create` produces a
    fresh sandbox and a running agent.
    """

    def __init__(
        self,
        *,
        config: OpenCodeConfig,
        sandbox_backend: SandboxBackend,
        mode: Literal["black_box", "transparent_proxy"] = "black_box",
        verifier: Verifier | None = None,
        install_timeout_s: int = 240,
        setup_timeout_s: int = 300,
    ) -> None:
        if mode not in {"black_box", "transparent_proxy"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._config = config
        self._backend = sandbox_backend
        self._mode = mode
        self._verifier = verifier
        self._install_timeout_s = install_timeout_s
        self._setup_timeout_s = setup_timeout_s

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> OpenCodeSession:
        oc_task = OpenCodeTask.coerce(task)
        sandbox_timeout = int(self._config.agent_timeout_s) + 300

        sandbox = self._backend.create(
            timeout_s=sandbox_timeout,
            metadata={"episode_id": episode_id} if episode_id else None,
        )
        try:
            self._bootstrap_sandbox(sandbox, oc_task)
        except Exception:
            sandbox.kill()
            raise

        base_url_override: str | None = None
        proxy_trace_path: str | None = None
        proxy_bg_job: BgJob | None = None
        if self._mode == "transparent_proxy":
            proxy_bg_job, base_url_override, proxy_trace_path = self._start_proxy(
                sandbox
            )
            # Rewrite opencode.json so opencode's configured baseURL is the
            # proxy, not the upstream. Force ``openai_compatible`` so opencode
            # routes through ``/v1/chat/completions`` (which the proxy serves)
            # rather than provider-specific endpoints like OpenAI's
            # ``/v1/responses`` (which ``@ai-sdk/openai`` uses and which vLLM
            # does not expose).
            from .config import OpenCodeConfig as _OCC

            proxy_cfg = _OCC(
                **{
                    **self._config.model_dump(),
                    "provider": "openai_compatible",
                    "base_url": base_url_override,
                }
            )
            sandbox.write_text(
                opencode_config_path(self._config),
                build_opencode_json(proxy_cfg),
            )

        session = OpenCodeSession(
            sandbox=sandbox,
            config=self._config,
            task=oc_task,
            verifier=self._verifier,
            base_url_override=base_url_override,
            proxy_trace_path=proxy_trace_path,
            proxy_bg_job=proxy_bg_job,
        )
        session.start_agent()
        return session

    # ------------------------------------------------------------------
    def _wait_for_sandbox_ready(
        self,
        sandbox: SandboxHandle,
        *,
        attempts: int = 15,
        delay_s: float = 1.0,
    ) -> None:
        """Probe the sandbox until it can execute a trivial command.

        E2B (and other backends) sometimes return a handle before the guest
        is fully ready. Issue ``echo ok`` with short timeouts until it
        succeeds — the Harbor/verifiers reference projects do something
        similar via ``wait_for_creation``. Returns silently on success;
        raises ``RuntimeError`` if the probe never succeeds.
        """
        import time

        last_err = ""
        for _ in range(attempts):
            try:
                r = sandbox.exec("echo ok", timeout=5)
                if r.exit_code == 0 and "ok" in (r.stdout or ""):
                    return
                last_err = (r.stderr or r.stdout or "").strip() or f"exit={r.exit_code}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(delay_s)
        raise RuntimeError(
            f"sandbox did not become ready within {attempts * delay_s:.0f}s "
            f"(last error: {last_err})"
        )

    def _exec_with_retry(
        self,
        sandbox: SandboxHandle,
        cmd: str,
        *,
        timeout: float,
        attempts: int = 3,
        backoff_s: float = 3.0,
        label: str = "cmd",
    ):
        """Run ``sandbox.exec`` with exponential backoff on transient failures.

        Transient is defined as ``exit_code != 0`` AND empty stderr (SIGKILL
        / network blip signature) OR an exception during exec. A final
        failure is raised as ``RuntimeError`` with the last exit code +
        stderr so the caller has something to debug with.
        """
        import time

        last_stdout = ""
        last_stderr = ""
        last_exit = 0
        for i in range(attempts):
            try:
                r = sandbox.exec(cmd, timeout=timeout)
                if r.exit_code == 0:
                    return r
                last_stdout = r.stdout or ""
                last_stderr = r.stderr or ""
                last_exit = r.exit_code
                # Non-empty stderr usually means the command failed for a
                # deterministic reason — don't retry, surface it.
                if last_stderr.strip():
                    break
            except Exception as exc:  # noqa: BLE001
                last_stderr = f"{type(exc).__name__}: {exc}"
                last_exit = -1
            if i + 1 < attempts:
                time.sleep(backoff_s * (2**i))
        raise RuntimeError(
            f"{label} failed after {attempts} attempts "
            f"(exit={last_exit}, stderr={last_stderr!r}, stdout_tail={last_stdout[-400:]!r})"
        )

    def _bootstrap_sandbox(
        self,
        sandbox: SandboxHandle,
        task: OpenCodeTask,
    ) -> None:
        """Install OpenCode, write config + task files, run optional setup."""

        # Stage 1: wait for the sandbox to be responsive.
        self._wait_for_sandbox_ready(sandbox)

        # Stage 2: install opencode with retries (curl | bash is flaky).
        self._exec_with_retry(
            sandbox,
            build_install_cmd(self._config),
            timeout=self._install_timeout_s,
            attempts=3,
            backoff_s=3.0,
            label="opencode install",
        )

        sandbox.write_text(
            opencode_config_path(self._config),
            build_opencode_json(self._config),
        )
        sandbox.write_text(instruction_path(self._config), task.instruction)

        if self._config.system_prompt:
            sandbox.write_text(
                system_prompt_path(self._config),
                self._config.system_prompt,
            )

        for remote_path, content in task.upload_files.items():
            sandbox.write_text(remote_path, content)

        if self._config.extra_setup_shell:
            self._exec_with_retry(
                sandbox,
                self._config.extra_setup_shell,
                timeout=self._setup_timeout_s,
                attempts=2,
                backoff_s=2.0,
                label="extra_setup_shell",
            )

        if task.setup_shell:
            r = sandbox.exec(task.setup_shell, timeout=self._setup_timeout_s)
            if r.exit_code != 0:
                raise RuntimeError(
                    f"task.setup_shell failed ({r.exit_code}): {r.stderr}"
                )

    def _start_proxy(
        self,
        sandbox: SandboxHandle,
    ) -> tuple[BgJob, str, str]:
        """Install proxy deps, start the proxy as a bg job inside the sandbox.

        Returns ``(proxy_bg_job, base_url_override, proxy_trace_path)``.
        """
        # Install proxy deps with retries. pip can transiently fail due to
        # PyPI connectivity or a sandbox not yet fully ready.
        self._exec_with_retry(
            sandbox,
            "pip install --quiet 'fastapi>=0.104' 'uvicorn[standard]>=0.24' "
            "'httpx>=0.27' 2>&1 | tail -20",
            timeout=180,
            attempts=3,
            backoff_s=2.0,
            label="proxy deps install",
        )

        # Upload the proxy module into the sandbox (standalone copy — avoids
        # depending on openenv.* inside the sandbox Python).
        proxy_src_path = Path(__file__).parent / "interception.py"
        sandbox.write_text(
            "/home/user/proxy/interception.py",
            proxy_src_path.read_text(),
        )
        sandbox.write_text("/home/user/proxy/__init__.py", "")

        cap_flag = ""
        if self._config.proxy_max_tokens_cap is not None:
            cap_flag = f"--max-tokens-cap {self._config.proxy_max_tokens_cap} "
        thinking_flag = ""
        if self._config.proxy_disable_thinking:
            thinking_flag = "--disable-thinking "
        # Force the upstream model id on every forwarded request. opencode's
        # internal title-generation call sometimes strips the provider prefix
        # (e.g. sends ``Qwen3.5-4B`` instead of ``Qwen/Qwen3.5-4B``) which the
        # upstream vLLM then rejects with a 404.
        model_override_flag = ""
        upstream_model = self._config.model.split("/", 1)[-1]
        if upstream_model:
            model_override_flag = f"--model-override '{upstream_model}' "
        proxy_cmd = (
            "cd /home/user/proxy && "
            "python interception.py "
            f"--upstream-url {self._config.base_url} "
            f"--upstream-api-key {self._config.api_key} "
            f"--trace {_PROXY_TRACE_PATH} "
            f"--port {_PROXY_PORT} "
            f"--top-logprobs {self._config.proxy_top_logprobs} "
            f"{cap_flag}"
            f"{thinking_flag}"
            f"{model_override_flag}"
            f"> {_PROXY_LOG_PATH} 2>&1"
        )
        proxy_job = sandbox.start_bg(proxy_cmd)

        # Wait for the proxy to start listening.
        for _ in range(40):
            r = sandbox.exec(
                f"curl -sf http://127.0.0.1:{_PROXY_PORT}/healthz",
                timeout=5,
            )
            if r.exit_code == 0:
                break
            import time

            time.sleep(0.25)
        else:
            log = ""
            try:
                log = sandbox.read_text(_PROXY_LOG_PATH)
            except Exception:
                pass
            proxy_job.kill()
            raise RuntimeError(
                f"proxy did not start within 10s. log:\n{log[-2000:]}"
            )

        base_url_override = f"http://127.0.0.1:{_PROXY_PORT}/v1"
        return proxy_job, base_url_override, _PROXY_TRACE_PATH


from pathlib import Path  # noqa: E402  (used only inside _start_proxy)


__all__ = [
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "Verifier",
]
