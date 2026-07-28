# SPDX-License-Identifier: BSD-3-Clause

"""The Harbor environment server.

Serves a directory of [Harbor](https://www.harborframework.com/docs/tasks) task
directories as a Gymnasium-style OpenEnv environment. The task directory is the
interface: the very same directory runs under `harbor run`, so a task set built
with [Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv) needs no conversion
to be trained against here.

This module is deliberately thin. It resolves which task to run
(`task.TaskCatalog`), asks a backend to execute things (`sandbox.Sandbox`), and
forwards the verifier's verdict (`reward.RewardReport`). It never computes a
reward itself.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment


# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from harbor_env.models import HarborAction, HarborObservation, HarborState
    from harbor_env.server.sandbox import (
        create_sandbox,
        ExecResult,
        resolve_within,
        Sandbox,
        SandboxError,
    )
    from harbor_env.server.task import HarborTask, resolve_task_source, TaskCatalog
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from models import HarborAction, HarborObservation, HarborState
    from sandbox import (
        create_sandbox,
        ExecResult,
        resolve_within,
        Sandbox,
        SandboxError,
    )
    from task import HarborTask, resolve_task_source, TaskCatalog


logger = logging.getLogger(__name__)

DEFAULT_COMMAND_TIMEOUT_S = 120.0

#: Upper bound on how many task ids [`HarborState.available_tasks`] carries, so
#: a catalog with thousands of tasks does not bloat every state response.
MAX_LISTED_TASKS = 200

_ActionHandler = Callable[[HarborAction, Sandbox, HarborTask], HarborObservation]


class HarborEnvironment(Environment[HarborAction, HarborObservation, HarborState]):
    """Runs Harbor task directories as an OpenEnv environment.

    Each `reset()` starts a fresh sandbox for one task; the agent then works
    through `exec` / `read` / `write` actions, and the training loop ends the
    episode with `evaluate`, which runs the task's own `tests/test.sh` and
    forwards whatever reward it wrote.

    Args:
        tasks (`str`, *optional*):
            Where tasks live: a local directory, or a Hugging Face dataset
            reference such as `hf://datasets/<org>/<name>`. Defaults to
            `$HARBOR_TASKS_DIR`, then to the bundled example tasks.
        mode (`str`, *optional*):
            Sandbox backend, `"local"` or `"docker"`. Defaults to `$HARBOR_MODE`,
            then `"local"`.
        default_task_id (`str`, *optional*):
            Task used when `reset()` is called without one. Defaults to
            `$HARBOR_DEFAULT_TASK_ID`.
        command_timeout_s (`float`, *optional*, defaults to `120.0`):
            Timeout applied to agent `exec` actions. The verifier and the oracle
            instead use the timeouts declared in `task.toml`.
        sandbox_factory (`Callable[[str], Sandbox]`, *optional*):
            Builds the sandbox for a mode. Injected by tests; defaults to
            `sandbox.create_sandbox`.

    Examples:

    ```python
    env = HarborEnvironment(tasks="./tasks", mode="docker")
    obs = env.reset(task_id="pallets__click-2951")
    env.step(HarborAction(action_type="write", path="src/click/core.py", content=fixed))
    result = env.step(HarborAction(action_type="evaluate"))
    result.reward  # whatever tests/test.sh wrote to /logs/verifier
    ```
    """

    # Every episode gets its own sandbox and its own working directory, so
    # sessions never share mutable state.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        tasks: str | None = None,
        mode: str | None = None,
        default_task_id: str | None = None,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        sandbox_factory: Callable[[str], Sandbox] | None = None,
    ) -> None:
        super().__init__()
        self.mode = (mode or os.getenv("HARBOR_MODE") or "local").lower()
        self.command_timeout_s = command_timeout_s
        self.default_task_id = (
            default_task_id or os.getenv("HARBOR_DEFAULT_TASK_ID") or None
        )
        self._sandbox_factory = sandbox_factory or self._default_sandbox_factory
        self.catalog = TaskCatalog(
            resolve_task_source(tasks or os.getenv("HARBOR_TASKS_DIR"))
        )

        self._task: HarborTask | None = None
        self._sandbox: Sandbox | None = None
        self._state = HarborState(mode=self.mode)
        self._refresh_catalog_state()

        self._handlers: dict[str, _ActionHandler] = {
            "exec": self._do_exec,
            "read": self._do_read,
            "write": self._do_write,
            "evaluate": self._do_evaluate,
            "solve": self._do_solve,
        }

    # --- Gymnasium API ------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> HarborObservation:
        """Start a fresh episode on one task.

        Args:
            seed (`int`, *optional*):
                Unused; Harbor tasks are deterministic by construction.
            episode_id (`str`, *optional*):
                Identifier recorded in the state.
            task_id (`str`, *optional*):
                Which task to run. Defaults to `default_task_id`, or to the only
                task in the catalog when there is exactly one.

        Returns:
            [`HarborObservation`] carrying the task's instruction.
        """
        del seed

        self.close()
        # A failed reset must not leave the previous episode's state behind.
        # close() has already torn the sandbox down, so anything read from
        # `state` after a raise below would describe a task that is no longer
        # running. Drop to "no episode" first, then do the work that can fail.
        self._state = HarborState(mode=self.mode)
        self._refresh_catalog_state()

        task = self.catalog.get(self._select_task_id(kwargs.get("task_id")))
        sandbox = self._sandbox_factory(self.mode)
        try:
            sandbox.start(task)
        except Exception:
            sandbox.close()
            raise

        self._task = task
        self._sandbox = sandbox
        self._state = HarborState(
            episode_id=episode_id or str(uuid4()),
            task_id=task.task_id,
            task_name=task.name,
            task_path=str(task.path),
            workdir=sandbox.paths.workdir,
            mode=sandbox.mode,
        )
        self._refresh_catalog_state()

        return self._observe(
            action_type="reset",
            output="",
            info={"task": task.summary(), "paths": sandbox.paths.as_env()},
        )

    def step(
        self,
        action: HarborAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> HarborObservation:
        """Apply one action to the running episode."""
        del timeout_s, kwargs

        if not isinstance(action, HarborAction):
            raise TypeError(f"expected HarborAction, got {type(action).__name__}")

        self._state.step_count += 1
        self._state.last_action_type = action.action_type

        try:
            sandbox, task = self._require_episode()
            return self._handlers[action.action_type](action, sandbox, task)
        except Exception as exc:
            # Invalid actions and sandbox failures end up in the observation so a
            # policy can recover; only server faults propagate.
            logger.info("action %s failed: %s", action.action_type, exc)
            return self._observe(
                action_type=action.action_type,
                output="",
                success=False,
                error=str(exc),
            )

    @property
    def state(self) -> HarborState:
        """Current episode state."""
        return self._state

    def close(self) -> None:
        """Tear down the running sandbox, if any."""
        if self._sandbox is not None:
            self._sandbox.close()
        self._sandbox = None
        self._task = None

    # --- action handlers ----------------------------------------------------

    def _do_exec(
        self, action: HarborAction, sandbox: Sandbox, task: HarborTask
    ) -> HarborObservation:
        del task
        if not action.command.strip():
            raise ValueError("exec requires a non-empty command")
        result = sandbox.exec(
            action.command,
            timeout_s=action.timeout_s or self.command_timeout_s,
            env=sandbox.task_env,
        )
        return self._from_exec("exec", result)

    def _do_read(
        self, action: HarborAction, sandbox: Sandbox, task: HarborTask
    ) -> HarborObservation:
        del task
        target = resolve_within(sandbox.paths.workdir, action.path)
        content = sandbox.read_text(target)
        if content is None:
            raise FileNotFoundError(f"no such file: {action.path}")
        return self._observe("read", output=content)

    def _do_write(
        self, action: HarborAction, sandbox: Sandbox, task: HarborTask
    ) -> HarborObservation:
        del task
        target = resolve_within(sandbox.paths.workdir, action.path)
        sandbox.write_text(target, action.content)
        return self._observe(
            "write", output=f"wrote {len(action.content)} bytes to {action.path}"
        )

    def _do_evaluate(
        self, action: HarborAction, sandbox: Sandbox, task: HarborTask
    ) -> HarborObservation:
        """Run the task's verifier and forward its reward. Ends the episode."""
        del action
        result = sandbox.run_verifier(task)
        report = sandbox.reward_report()

        self._state.evaluated = True
        self._state.reward = report.value
        self._state.last_exit_code = result.exit_code

        info = {**report.as_info(), "verifier_exit_code": result.exit_code}
        if not report.graded:
            # No reward file means the episode cannot be scored. Say so instead of
            # inventing a number that would be indistinguishable from a real 0.0.
            error = (
                "the verifier wrote no reward file to "
                f"{sandbox.paths.logs_verifier}; the episode cannot be scored"
            )
            hint = sandbox.explain_missing_reward(task)
            if hint:
                error = f"{error}. {hint}"
            return self._observe(
                "evaluate",
                output=result.output,
                success=False,
                error=error,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                done=True,
                info=info,
            )
        return self._observe(
            "evaluate",
            output=result.output,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            reward=report.value,
            done=True,
            info=info,
        )

    def _do_solve(
        self, action: HarborAction, sandbox: Sandbox, task: HarborTask
    ) -> HarborObservation:
        """Apply the task's reference solution — Harbor's oracle agent.

        Orchestration-only: it validates that a task is solvable and produces gold
        trajectories. Follow it with `evaluate`, which should return the task's
        maximum reward.
        """
        del action
        result = sandbox.run_solution(task)
        return self._from_exec("solve", result)

    # --- observation plumbing -----------------------------------------------

    def _from_exec(self, action_type: str, result: ExecResult) -> HarborObservation:
        self._state.last_exit_code = result.exit_code
        error = ""
        if result.timed_out:
            error = "command timed out"
        return self._observe(
            action_type,
            output=result.output,
            success=result.ok,
            error=error,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
        )

    def _observe(
        self,
        action_type: str,
        output: str,
        success: bool = True,
        error: str = "",
        exit_code: int | None = None,
        timed_out: bool = False,
        reward: float | None = None,
        done: bool = False,
        info: dict[str, Any] | None = None,
    ) -> HarborObservation:
        return HarborObservation(
            instruction=self._task.instruction if self._task else "",
            output=output,
            action_type=action_type,
            success=success,
            error=error,
            exit_code=exit_code,
            timed_out=timed_out,
            task_id=self._state.task_id,
            task_name=self._state.task_name,
            workdir=self._state.workdir,
            mode=self._state.mode,
            info=info or {},
            reward=reward,
            done=done,
        )

    # --- helpers ------------------------------------------------------------

    def _default_sandbox_factory(self, mode: str) -> Sandbox:
        if mode == "docker":
            image = os.getenv("HARBOR_DEFAULT_IMAGE")
            return create_sandbox(mode, **({"default_image": image} if image else {}))
        return create_sandbox(mode)

    def _select_task_id(self, requested: str | None) -> str:
        task_id = requested or self.default_task_id
        if task_id:
            return task_id
        available = self.catalog.task_ids()
        if len(available) == 1:
            return available[0]
        raise ValueError(
            "reset() needs a task_id: the catalog at "
            f"{self.catalog.root} holds {len(available)} tasks. "
            f"Available: {available[:10]}{' ...' if len(available) > 10 else ''}"
        )

    def _require_episode(self) -> tuple[Sandbox, HarborTask]:
        if self._sandbox is None or self._task is None:
            raise SandboxError("no episode is running; call reset() first")
        return self._sandbox, self._task

    def _refresh_catalog_state(self) -> None:
        task_ids = self.catalog.task_ids()
        self._state.task_count = len(task_ids)
        self._state.available_tasks = task_ids[:MAX_LISTED_TASKS]
