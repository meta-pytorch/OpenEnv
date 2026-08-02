"""Drive rollouts without a server: boot capture, run tasks, tear down.

`openenv harbor rollout` uses this. It exists so the whole path (LLM, capture proxy, forwarding,
seam, Harbor trial, sandbox, verifier, reconciliation) can be exercised with no env server in
the way. When something breaks, that halves the search space immediately: if this works and `serve` does
not, the problem is the serving layer and nothing below it.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path
from typing import Any

from openenv.core.harness.capture.server import create_app

from .models import HarborRolloutResult
from .rollout import run_rollout
from .tasks import resolve_task_dirs


def _require_free_port(port: int) -> None:
    """Raise if anything is already listening on `port`, naming the holder when we can find it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(
                f"capture port :{port} is already in use ({exc.strerror}). {_port_holder(port)}"
                " Stop it or pass a different port: a second server on this port cannot bind, and "
                "the agent would silently talk to the older one."
            ) from exc


def _port_holder(port: int) -> str:
    """Best-effort description of the process holding `port`, for the error message only."""
    import shutil
    import subprocess

    if not shutil.which("ss"):
        return ""
    with contextlib.suppress(Exception):
        out = subprocess.run(
            ["ss", "-ltnp"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if f":{port} " in line and "users:" in line:
                return f"Held by {line.split('users:', 1)[1].strip()}."
    return ""


def _health_instance(port: int) -> str | None:
    """Instance id reported by whatever is serving `port`, or `None` if nothing answers yet."""
    import httpx

    with contextlib.suppress(Exception):
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        if resp.status_code == 200:
            return str(resp.json().get("instance") or "unknown")
    return None


class CaptureServer:
    """The capture proxy, running in a background thread for the life of a batch.

    A thread rather than a subprocess because the rollout path needs the live `SessionRegistry` — it
    mints a session, then reads the graph back out of it directly. Going through HTTP for that would
    add a serialisation round trip and a failure mode for no benefit.
    """

    def __init__(
        self,
        *,
        llm_url: str,
        model: str,
        port: int = 8100,
        max_output_tokens: int = 8192,
    ) -> None:
        self.app = create_app(
            llm_url=llm_url, model=model, max_output_tokens=max_output_tokens
        )
        self.port = port
        self._thread: threading.Thread | None = None
        self._server: Any = None

    @property
    def registry(self) -> Any:
        return self.app.state.registry

    def start(self, timeout_s: float = 30.0) -> None:
        """Bind the port and confirm that the process answering on it is *this* one.

        Raises:
            RuntimeError:
                If the port is already held, or if the server that comes up on it is not ours.
        """
        import uvicorn

        # Fail before uvicorn does. Its bind error surfaces on a background thread, where nothing
        # observes it, and the port stays served by whoever holds it.
        _require_free_port(self.port)

        config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Reachability is not identity. A stale process on this port answers every probe, so the
        # check is that /health reports our own instance id.
        want = self.app.state.instance_id
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._thread.is_alive():
                raise RuntimeError(
                    f"capture server thread exited while starting on :{self.port} "
                    "(most likely the port was taken between the check and the bind)"
                )
            got = _health_instance(self.port)
            if got == want:
                return
            if got is not None:
                raise RuntimeError(
                    f"port :{self.port} is served by a different capture server (instance {got}, "
                    f"expected {want}). Stop the process holding it, or pass a different port; "
                    "sessions minted here would be rejected there and every rollout would see "
                    "no model calls."
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"capture server did not come up on :{self.port} within {timeout_s:.0f}s"
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)


async def run_batch(
    *,
    llm_url: str,
    dataset: str,
    task_indices: list[int],
    harness: str = "opencode",
    sandbox: str = "e2b",
    model: str | None = None,
    port: int = 8100,
    expose: str = "gradio",
    trials_dir: Path | None = None,
    reward_key: str = "",
    keep_sandbox: bool = False,
    force_build: bool = False,
    env_file: str | None = None,
) -> list[HarborRolloutResult]:
    """Run `task_indices` from `dataset` and print a per-rollout report.

    Args:
        llm_url (`str`):
            OpenAI-spec inference endpoint.
        dataset (`str`):
            Dataset spec (HF repo id, local dir, or Harbor `name@version`).
        task_indices (`list[int]`):
            Which tasks to run, by index into the resolved dataset.
        harness (`str`, *optional*, defaults to `"opencode"`):
            Seam name or `module:Class`.
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Harbor environment type.
        expose (`str`, *optional*, defaults to `"gradio"`):
            How the sandbox reaches the capture proxy: `gradio`, `cloudflare` or `direct`.

    Returns:
        `list[HarborRolloutResult]`: One per index, in order.
    """
    # Imported here, not at module scope: a hosted deployment mounts the capture proxy on its
    # own app and never forwards, so `forwarding` is not shipped there. `serving` imports
    # `CaptureServer` from this module, and a module-level import would break that.
    from openenv.core.harness.capture.forwarding import make_forwarder

    from .startup import prepare

    caps = prepare(
        llm_url=llm_url,
        model=model,
        datasets=[dataset],
        env_file=env_file,
        require_llm=True,
        quiet=False,
    )
    model = caps.llm.get("model") or model or ""

    if sandbox not in caps.available_sandboxes:
        detail = next(
            (s.detail for s in caps.sandboxes if s.name == sandbox), "not checked"
        )
        raise RuntimeError(f"sandbox {sandbox!r} is not usable here: {detail}")

    task_dirs = resolve_task_dirs(dataset)
    trials_dir = trials_dir or Path("/tmp/openenv-harbor-trials")
    trials_dir.mkdir(parents=True, exist_ok=True)

    capture = CaptureServer(llm_url=llm_url, model=model, port=port)
    capture.start()
    forwarder = make_forwarder(expose)
    public_url = forwarder.start(port)
    print(f"\ncapture  :{port} -> {public_url}  ({forwarder.name})")
    print(f"trials   {trials_dir}\n")

    results: list[HarborRolloutResult] = []
    try:
        for i in task_indices:
            if not 0 <= i < len(task_dirs):
                print(f"  skip index {i}: out of range (dataset has {len(task_dirs)})")
                continue
            task_dir = task_dirs[i]
            print(f"[{harness} / {sandbox}] task {i}: {task_dir.name} ...", flush=True)
            result = await run_rollout(
                task_dir=task_dir,
                harness=harness,
                sandbox=sandbox,
                registry=capture.registry,
                intercept_url=public_url,
                model=model,
                trials_dir=trials_dir,
                dataset=dataset,
                reward_key=reward_key,
                keep_sandbox=keep_sandbox,
                force_build=force_build,
            )
            results.append(result)
            print("   " + _summarise(result))
            for finding in result.findings[:3]:
                print(f"      {finding[:150]}")
    finally:
        forwarder.stop()
        capture.stop()

    print("\n" + _report(results))
    return results


def _summarise(r: HarborRolloutResult) -> str:
    reward = "None" if r.reward is None else f"{r.reward:.2f}"
    mode = "multi-turn" if r.multi_turn else "per-turn"
    status = "ok" if r.ok else f"FAILED ({r.exception_type or 'error'})"
    return (
        f"{status:<26} reward={reward:<6} turns={r.n_turns:<3} roots={r.n_roots:<3} "
        f"{mode:<11} tokens={r.n_trainable_tokens:<6} atif={r.atif:<9} {r.wall_s:.0f}s"
        + (f"\n      {r.error[:180]}" if r.error else "")
    )


def _report(results: list[HarborRolloutResult]) -> str:
    if not results:
        return "no rollouts ran"
    ok = sum(1 for r in results if r.ok)
    graded = [r for r in results if r.reward is not None]
    solved = sum(1 for r in graded if r.reward and r.reward > 0)
    lines = [
        "=" * 78,
        f"capture   {ok}/{len(results)} usable",
        f"solved    {solved}/{len(graded)} graded"
        + (
            f"   ({len(results) - len(graded)} ungraded — the verifier never ran)"
            if len(graded) != len(results)
            else ""
        ),
        f"tokens    {sum(r.n_trainable_tokens for r in results)} trainable across "
        f"{sum(r.n_turns for r in results)} turns",
    ]
    # Capture quality and task success are independent, and conflating them has burned us before:
    # a perfectly captured rollout can score 0 because the model was wrong.
    lines.append("NOTE: capture and reward are independent measurements.")
    return "\n".join(lines)
