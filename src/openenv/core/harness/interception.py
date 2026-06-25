"""InterceptionServer: an OpenAI-compatible HTTP proxy that gates every LLM call from a harness.

A harness (an agent that owns its loop, running outside the trainer) points its OpenAI base URL at
`/rollout/{id}/v1`. Each chat-completions call blocks here until the rollout worker generates and
delivers a response. One instance multiplexes many rollouts by `rollout_id`. This is the on-policy
capture seam for training agentic harnesses (see `rollout_worker.HarnessRolloutWorker`).

Note: a richer interception/sandbox stack is proposed in PR #694. This is a clean, minimal core
implementation focused on the transport. The two should be reconciled before merge.
"""

from __future__ import annotations

import asyncio
import queue as _queue
import threading
import uuid

from aiohttp import web


def _resolve_if_pending(fut) -> None:
    """Unblock a waiting chat-completions handler with an empty stop response (teardown path)."""
    if not fut.done():
        fut.set_result(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ]
            }
        )


class InterceptionServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host, self._req_port = host, port
        self.port = port
        self._loop = asyncio.new_event_loop()
        self._runner: web.AppRunner | None = None
        self._intercepts: dict[str, dict] = {}
        self._queues: dict[str, _queue.Queue] = {}

    def register_rollout(self, rollout_id: str) -> _queue.Queue:
        q: _queue.Queue = _queue.Queue()
        self._queues[rollout_id] = q
        return q

    def unregister_rollout(self, rollout_id: str) -> None:
        self._queues.pop(rollout_id, None)
        for rid, it in [
            (k, v) for k, v in self._intercepts.items() if v["rollout_id"] == rollout_id
        ]:
            fut = it.get("future")
            if fut is not None:
                # Resolve any in-flight request so a blocked agent does not hang until its timeout
                # (e.g. if generate() failed mid-rollout and the session is being torn down).
                self._loop.call_soon_threadsafe(_resolve_if_pending, fut)
            self._intercepts.pop(rid, None)

    def get_intercept(self, request_id: str) -> dict | None:
        return self._intercepts.get(request_id)

    def deliver(self, request_id: str, content: str) -> None:
        intercept = self._intercepts.get(request_id)
        if not intercept:
            return
        resp = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
        self._loop.call_soon_threadsafe(intercept["future"].set_result, resp)

    async def _chat(self, request: web.Request) -> web.Response:
        rollout_id = request.match_info["rollout_id"]
        body = await request.json()
        request_id = uuid.uuid4().hex  # full uuid: no collision risk for long runs
        intercept = {
            "request_id": request_id,
            "rollout_id": rollout_id,
            "messages": body.get("messages"),
            "tools": body.get("tools"),
            "future": self._loop.create_future(),
        }
        self._intercepts[request_id] = intercept
        self._queues[rollout_id].put_nowait(request_id)
        try:
            return web.json_response(await intercept["future"])
        finally:
            self._intercepts.pop(request_id, None)

    async def _exit(self, request: web.Request) -> web.Response:
        q = self._queues.get(request.match_info["rollout_id"])
        if q is not None:
            q.put_nowait(None)  # sentinel: agent done
        return web.json_response({"ok": True})

    def start(self) -> None:
        ready = threading.Event()
        err: dict[str, BaseException] = {}

        async def _run():
            try:
                app = web.Application()
                app.router.add_post(
                    "/rollout/{rollout_id}/v1/chat/completions", self._chat
                )
                app.router.add_post("/rollout/{rollout_id}/exit", self._exit)
                self._runner = web.AppRunner(app)
                await self._runner.setup()
                site = web.TCPSite(self._runner, self.host, self._req_port)
                await site.start()
                sock = next(iter(site._server.sockets))  # type: ignore[attr-defined]
                self.port = sock.getsockname()[1]
            except BaseException as e:  # noqa: BLE001
                err["e"] = e
            finally:
                ready.set()

        self._loop.create_task(_run())
        threading.Thread(
            target=self._loop.run_forever, daemon=True, name="interception"
        ).start()
        if not ready.wait(timeout=10):
            raise RuntimeError("InterceptionServer did not start within 10s")
        if "e" in err:
            raise RuntimeError(
                f"InterceptionServer failed to start: {err['e']}"
            ) from err["e"]
        if not self.port:
            raise RuntimeError("InterceptionServer started but no port was bound")

    def stop(self) -> None:
        async def _cleanup():
            if self._runner is not None:
                await self._runner.cleanup()

        try:
            fut = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
            fut.result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
