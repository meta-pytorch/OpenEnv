"""The OpenEnv environment: Task API for discovery, one MCP tool for execution.

`run_rollout` is a single long tool call rather than `reset`/`step`, because OpenEnv's HTTP handlers
construct and close an environment per request while a Harbor rollout is one stateful 60-600s run
that the harness drives. There is no meaningful `step()` to expose while opencode drives itself.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation

from .models import HarborState

# A rollout is minutes, not seconds. OpenEnv's MCP tools default to 30s and there is no config knob,
# so the only way to raise it is to shadow `step`/`step_async` and inject a default.
_ROLLOUT_TIMEOUT_S = 1800.0


class HarborEnvironment(MCPEnvironment):
    """Per-session environment exposing `run_rollout` and `capabilities` over MCP."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Server-wide config, set once by `serving.build_app`. Class-level because OpenEnv builds a
    # throwaway instance per `/metadata` and `/schema` request, and `__init__` must stay cheap and
    # credential-free or the docs page cannot load.
    _datasets: list[str] = []
    _llm_url: str = ""
    _model: str = ""
    # The validated LLM report from startup. Rebuilding it per request would mean re-probing the
    # endpoint on every `capabilities()` call, so the startup verdict (including `ok`) is carried.
    _llm: dict[str, Any] = {}

    @classmethod
    def configure(
        cls,
        *,
        datasets: list[str],
        llm_url: str = "",
        model: str = "",
        llm: dict[str, Any] | None = None,
    ) -> None:
        cls._datasets = list(datasets)
        cls._llm_url = llm_url
        cls._model = model
        cls._llm = dict(llm or {})

    def __init__(self) -> None:
        from fastmcp import FastMCP

        from .capabilities import capabilities as _capabilities
        from .tasks import HarborTaskProvider

        self._provider = HarborTaskProvider(self._datasets)
        self._state = HarborState(episode_id=str(uuid4()), llm_url=self._llm_url)

        mcp = FastMCP("harbor_env")

        @mcp.tool
        def run_rollout(
            split: str = "",
            task_index: int = 0,
            harness: str = "opencode",
            sandbox: str = "e2b",
            reward_key: str = "",
            keep_sandbox: bool = False,
            force_build: bool = False,
        ) -> str:
            """Run one Harbor rollout and return a JSON `HarborRolloutResult`.

            `harness` and `sandbox` are per-call, so consecutive rollouts can use different agents
            and different backends against the same server.
            """
            return self._run_rollout(
                split,
                task_index,
                harness,
                sandbox,
                reward_key,
                keep_sandbox,
                force_build,
            )

        @mcp.tool
        def capabilities() -> str:
            """Harnesses, sandboxes, datasets and LLM status for this server."""
            caps = _capabilities(
                datasets=self._datasets,
                llm=self._llm or {"url": self._llm_url, "model": self._model},
            )
            return json.dumps(caps.to_dict())

        @mcp.tool
        def list_tasks(split: str = "", start: int = 0, stop: int = 20) -> str:
            """A window of tasks in a split, for browsing without pulling all of them."""
            return json.dumps(self._provider.get_task_range(split, start, stop))

        super().__init__(mcp)

    # --- Task API (OpenEnv discovers these by duck typing) ----------------
    def list_splits(self) -> list[dict[str, Any]]:
        return self._provider.list_splits()

    def num_tasks(self, split: str) -> int:
        return self._provider.num_tasks(split)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self._provider.list_tasks(split)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self._provider.get_task(split, index)

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        return self._provider.get_task_range(split, start, stop)

    # --- Environment ------------------------------------------------------
    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **_: Any
    ) -> Observation:
        """New episode. Boots nothing — a sandbox is created per `run_rollout`, not per reset."""
        service = self._service()
        self._state = HarborState(
            episode_id=episode_id or str(uuid4()),
            llm_url=self._llm_url,
            intercept_url=getattr(service, "public_url", "") if service else "",
        )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "status": "ready",
                "message": "Call run_rollout(split=..., task_index=..., harness=..., sandbox=...)",
                "datasets": self._datasets,
            },
        )

    def _step_impl(
        self, action: Any, timeout_s: float | None = None, **_: Any
    ) -> Observation:
        return Observation(
            done=False,
            reward=None,
            metadata={
                "error": f"Unknown action {type(action).__name__}; "
                "use CallToolAction(name='run_rollout', ...)"
            },
        )

    def step(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return super().step(action, timeout_s=timeout_s or _ROLLOUT_TIMEOUT_S, **kwargs)

    async def step_async(
        self, action: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return await super().step_async(
            action, timeout_s=timeout_s or _ROLLOUT_TIMEOUT_S, **kwargs
        )

    @property
    def state(self) -> HarborState:
        return self._state

    # --- internals --------------------------------------------------------
    @staticmethod
    def _service() -> Any:
        from .serving import HarborService

        return HarborService.current()

    def _run_rollout(
        self,
        split: str,
        task_index: int,
        harness: str,
        sandbox: str,
        reward_key: str,
        keep_sandbox: bool,
        force_build: bool,
    ) -> str:
        import asyncio
        from pathlib import Path

        from .models import HarborRolloutResult
        from .rollout import run_rollout as _run

        service = self._service()
        if service is None:
            return HarborRolloutResult(
                ok=False,
                error="server not initialised: no capture proxy is running",
                harness=harness,
                sandbox=sandbox,
            ).model_dump_json()

        split = split or (self._datasets[0] if self._datasets else "")
        try:
            task_dir = self._provider.task_dir(split, int(task_index))
        except Exception as exc:  # noqa: BLE001
            return HarborRolloutResult(
                ok=False, error=str(exc)[:400], harness=harness, sandbox=sandbox
            ).model_dump_json()

        result = asyncio.run(
            _run(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                registry=service.capture.registry,
                intercept_url=service.public_url,
                model=service.model,
                trials_dir=Path("/tmp/openenv-harbor-trials"),
                dataset=split,
                reward_key=reward_key,
                keep_sandbox=keep_sandbox,
                force_build=force_build,
            )
        )

        self._state.rollouts_completed += 1
        self._state.last_reward = result.reward
        self._state.last_task_id = result.task_id
        self._state.last_trial_name = result.trial_name
        return result.model_dump_json()
