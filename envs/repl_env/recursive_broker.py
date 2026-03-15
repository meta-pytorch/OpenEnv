# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Broker abstractions for blocking recursive calls.

This module implements the core Daytona-style broker contract in a framework-
independent way:
- `enqueue` submits a request and blocks for its response
- `pending` long-polls for outstanding requests
- `respond` completes a pending request

The current server path still uses an in-process worker, but these primitives
let the broker be driven by an external poller or HTTP transport later without
changing the environment or runner layers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Callable


@dataclass
class BrokerRequest:
    request_id: int
    kind: str
    prompts: list[str]
    model: str | None
    done: threading.Event = field(default_factory=threading.Event)
    result: list[str] | None = None
    error: str | None = None
    claimed: bool = False


@dataclass
class PendingBrokerRequest:
    request_id: int
    kind: str
    prompts: list[str]
    model: str | None


class InProcessRecursiveBroker:
    """Queue-based broker supporting both in-process workers and poll/respond."""

    def __init__(
        self,
        query_handler: Callable[[str, str | None], str],
        query_batched_handler: Callable[[list[str], str | None], list[str]],
        *,
        start_worker: bool = True,
    ) -> None:
        self.query_handler = query_handler
        self.query_batched_handler = query_batched_handler
        self._ids = count(1)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._requests: dict[int, BrokerRequest] = {}
        self._closed = False
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def health(self) -> dict[str, object]:
        with self._lock:
            pending_count = sum(
                1
                for request in self._requests.values()
                if request.result is None and request.error is None
            )
            return {
                "status": "ok",
                "pending_count": pending_count,
                "closed": self._closed,
            }

    def query(self, prompt: str, model: str | None = None) -> str:
        request = self._submit("single", [prompt], model)
        if request.error is not None:
            return f"Error: {request.error}"
        return request.result[0] if request.result else ""

    def query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        request = self._submit("batched", prompts, model)
        if request.error is not None:
            return [f"Error: {request.error}" for _ in prompts]
        return request.result or []

    def enqueue(
        self,
        kind: str,
        prompts: list[str],
        model: str | None = None,
        *,
        timeout_s: float | None = None,
    ) -> BrokerRequest:
        request = BrokerRequest(
            request_id=next(self._ids),
            kind=kind,
            prompts=list(prompts),
            model=model,
        )
        with self._condition:
            if self._closed:
                request.error = "broker is closed"
                request.done.set()
                return request
            self._requests[request.request_id] = request
            self._condition.notify_all()
        completed = request.done.wait(timeout=timeout_s)
        if not completed:
            with self._condition:
                current = self._requests.pop(request.request_id, None)
                if (
                    current is not None
                    and current.error is None
                    and current.result is None
                ):
                    current.error = f"request timed out after {timeout_s:.3f}s"
                    current.done.set()
                    return current
        return request

    def pending(self, timeout_s: float = 0.0) -> list[PendingBrokerRequest]:
        deadline = time.time() + timeout_s
        with self._condition:
            while True:
                pending = [
                    PendingBrokerRequest(
                        request_id=request.request_id,
                        kind=request.kind,
                        prompts=list(request.prompts),
                        model=request.model,
                    )
                    for request in self._requests.values()
                    if request.result is None
                    and request.error is None
                    and not request.claimed
                ]
                if pending or timeout_s <= 0 or self._closed:
                    for item in pending:
                        self._requests[item.request_id].claimed = True
                    return pending
                remaining = deadline - time.time()
                if remaining <= 0:
                    return []
                self._condition.wait(timeout=min(remaining, 1.0))

    def respond(
        self,
        request_id: int,
        *,
        result: list[str] | None = None,
        error: str | None = None,
    ) -> bool:
        with self._condition:
            request = self._requests.get(request_id)
            if request is None:
                return False
            request.result = result
            request.error = error
            request.done.set()
            self._condition.notify_all()
            return True

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for request in self._requests.values():
                if request.error is None and request.result is None:
                    request.error = "broker is closed"
                    request.done.set()
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            pending = self.pending(timeout_s=0.5)
            if self._closed and not pending:
                return
            for request in pending:
                try:
                    if request.kind == "single":
                        result = [self.query_handler(request.prompts[0], request.model)]
                    else:
                        result = self.query_batched_handler(
                            request.prompts, request.model
                        )
                    self.respond(request.request_id, result=result)
                except Exception as exc:
                    self.respond(request.request_id, error=str(exc))

    def _submit(
        self,
        kind: str,
        prompts: list[str],
        model: str | None,
    ) -> BrokerRequest:
        request = self.enqueue(kind, prompts, model)
        with self._condition:
            self._requests.pop(request.request_id, None)
        return request
