"""The upstream leg: one HTTP client to a vLLM OpenAI-compatible server.

vLLM only, deliberately. SGLang cannot support this layer at all — none of
`return_tokens_as_token_ids`, `logprobs_mode`, `processed_logprobs` or `return_token_ids` exist in
its tree, and its chat route returns token *text* with no ids (sgl-project/sglang#18378 requests
exactly this, for the same train/inference consistency reason). Carrying a two-engine abstraction for
a backend that structurally cannot work would be pretending we have a choice.

Two request params do all the work, and both are easy to get subtly wrong:

    return_token_ids=True   makes vLLM emit `response.prompt_token_ids` and `choice.token_ids`.
                            `prompt_token_ids` is the load-bearing one: it is the engine's own
                            tokenisation of the whole conversation so far, which is what lets turn
                            k+1 be matched against turn k by exact token prefix without us ever
                            tokenising locally.
    top_logprobs=0          must be SET, not omitted. vLLM only populates `logprobs.content[]` when
                            `top_logprobs` is not None, even with `logprobs=True`. Zero returns just
                            the sampled token's logprob, which is all training needs.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class UpstreamError(RuntimeError):
    """Any failure talking to the engine. Never leaks httpx types to callers."""


class UpstreamHTTPError(UpstreamError):
    """Engine answered with a non-2xx status."""

    def __init__(
        self, status_code: int, body: dict[str, Any] | str | None = None
    ) -> None:
        self.status_code = status_code
        self.body = body
        detail = body
        if isinstance(body, dict):
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error or body
        super().__init__(f"upstream returned {status_code}: {str(detail)[:400]}")


class UpstreamTimeoutError(UpstreamError):
    """Engine did not answer within the liveness ceiling."""


class UpstreamTransportError(UpstreamError):
    """Connection-level failure: refused, reset, DNS."""


def normalise_engine_base(url: str) -> str:
    """The engine root, with any trailing `/v1` removed.

    Every route this module builds is already `/v1/...`, so a caller who passes the OpenAI-style
    base (`http://host:8000/v1`, which is what most SDKs and most people hand you) would otherwise
    get `/v1/v1/chat/completions` and see a healthy engine reported as unreachable. Accept both
    forms and normalise here rather than making every call site remember which one it holds.
    """
    base = url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def prepare_request(
    request: dict[str, Any], *, served_model: str | None = None
) -> dict[str, Any]:
    """Add the params that make a response capturable. Mutates and returns `request`.

    Only the top level of `request` is mutated. Nested message dicts are copied before they are
    rewritten, because the caller's messages are the same objects the graph stores: rewriting them
    in place would mean a captured turn no longer records what the harness actually sent.
    """
    request["logprobs"] = True
    request["return_token_ids"] = True
    # Not `setdefault`: that keeps an explicitly-provided `None`, and vLLM only fills
    # `logprobs.content[]` when `top_logprobs` is not None. A harness that sends
    # `"top_logprobs": null` would then get a normal-looking response with no logprobs at all, so
    # every turn it produced would be silently untrainable.
    if request.get("top_logprobs") is None:
        request["top_logprobs"] = 0

    # vLLM reads a prior turn's thinking from `reasoning`, while the dialect transformers emit the
    # canonical `reasoning_content`. Without this rename an earlier turn's interleaved thinking
    # renders as an empty `<think></think>` and the prompt silently differs from what the model
    # actually produced — which breaks prefix matching for the turn after it.
    messages = request.get("messages")
    if isinstance(messages, list):
        rewritten: list[Any] = []
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("reasoning_content") is not None
            ):
                message = dict(message)
                message["reasoning"] = message.pop("reasoning_content")
            rewritten.append(message)
        request["messages"] = rewritten

    # `_served_model` is an internal marker the dialect transformers read; the server sets it before
    # transforming and the transformer strips it. Setting it here would be too late to be read and
    # would leak an unknown field to the engine, so this only guarantees it is gone.
    request.pop("_served_model", None)
    return request


def normalize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Canonicalise vLLM's response shape in place."""
    choices = response.get("choices")
    if not isinstance(choices, list):
        return response

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        message = choice.get("message")
        if isinstance(message, dict):
            if (
                message.get("reasoning_content") is None
                and message.get("reasoning") is not None
            ):
                message["reasoning_content"] = message.pop("reasoning")

        # Copy each token id onto its logprob entry. Not load-bearing — capture reads
        # `choice.token_ids` and the per-entry `logprob` — but it keeps a stored trace one shape,
        # so a consumer never has to know which engine produced it. Guarded on equal length because
        # a mismatch means the two lists are not describing the same tokens, and pairing them anyway
        # would silently attach the wrong id to every logprob.
        token_ids = choice.get("token_ids")
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        if isinstance(token_ids, list) and len(token_ids) == len(entries):
            for token_id, entry in zip(token_ids, entries):
                if isinstance(entry, dict):
                    entry.setdefault("token_id", token_id)
    return response


class InferenceClient:
    """Async client to one engine. One instance per server, shared across sessions."""

    # A high ceiling, not a per-request budget. Callers impose their own deadline; this exists only
    # so a wedged engine cannot pin a connection forever.
    _LIVENESS_TIMEOUT_S = 900.0
    _CONNECT_TIMEOUT_S = 30.0

    def __init__(self, base_url: str, *, served_model: str | None = None) -> None:
        self.base_url = normalise_engine_base(base_url)
        self.served_model = served_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    self._LIVENESS_TIMEOUT_S, connect=self._CONNECT_TIMEOUT_S
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """One non-streaming chat completion, prepared for capture and normalised on the way back."""
        body = prepare_request(dict(request), served_model=self.served_model)
        payload = await self._post("/v1/chat/completions", body)
        return normalize_response(payload)

    async def list_models(self) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        return response.json()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.post(path, json=body)
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc
        await self._raise_for_status(response)
        return response.json()

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        content = await response.aread()
        await response.aclose()
        body: dict[str, Any] | str | None = None
        text = content.decode("utf-8", errors="replace").strip()
        if text:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = text
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError(f"engine timed out: {exc}")
        return UpstreamTransportError(f"could not reach engine: {exc}")
