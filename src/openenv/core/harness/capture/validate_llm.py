"""Certify that an inference engine can actually support token-level capture.

This runs BEFORE a server binds a port, and refusing here is the entire point.

An engine missing `--return-tokens-as-token-ids --logprobs-mode processed_logprobs` still answers
every request perfectly well: it returns text, a `200`, and plausible-looking usage. What it does not
return is token ids. Every row rebuilt downstream is then empty, training silently does nothing, and
the first symptom is a loss curve that never moves days later. That failure has no loud edge, so the
check has to be up front and fatal.

Only vLLM implements the contract today. SGLang has none of the four capture knobs
(`return_tokens_as_token_ids`, `logprobs_mode`, `processed_logprobs`, `return_token_ids` all match
zero files in its tree) and its chat route returns token *text* only — see sgl-project/sglang#18378,
which requests exactly this and is motivated by the same train/inference consistency problem. A
hosted alternative exists but is narrow: fireworks-ai via the HF router honours vLLM's
`return_token_ids`, though every one of its live models is a reasoning model whose reasoning tokens
are dropped from history, so multi-turn stitching degrades to per-turn.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .validate import check_upstream_response


@dataclass
class LLMReport:
    """Outcome of certification. `ok` gates whether a server may start."""

    ok: bool
    llm_url: str
    model: str
    findings: list[str] = field(default_factory=list)
    n_prompt_ids: int = 0
    n_completion_ids: int = 0
    served_models: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return (
                f"engine OK: {self.n_completion_ids} completion ids, "
                f"{self.n_prompt_ids} prompt ids"
            )
        return "engine NOT usable for capture:\n  " + "\n  ".join(self.findings)


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def list_models(llm_url: str, timeout: float = 30.0) -> list[str]:
    """Served model ids, or [] if the endpoint is unreachable."""
    try:
        with urllib.request.urlopen(
            f"{llm_url.rstrip('/')}/v1/models", timeout=timeout
        ) as r:
            return [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
    except Exception:  # noqa: BLE001 - unreachable is reported by the caller, not raised here
        return []


def validate_llm(llm_url: str, model: str, *, timeout: float = 120.0) -> LLMReport:
    """Send one real completion and assert the response carries what capture needs.

    Deliberately a live probe rather than a flag inspection: launch flags are not readable over the
    API, and an engine can be started with the right arguments and still not behave (wrong version,
    a proxy in between that strips fields). The only trustworthy check is asking for a completion and
    looking at what comes back.
    """
    base = llm_url.rstrip("/")
    served = list_models(base, timeout=min(timeout, 30.0))

    if not served:
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            findings=[
                f"GET {base}/v1/models returned nothing; the engine is unreachable"
            ],
        )

    if model not in served:
        # Worth failing on rather than warning: a mismatched name is silently accepted by some
        # servers and then every request 404s at rollout time instead of at startup.
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            served_models=served,
            findings=[f"model {model!r} is not served here; available: {served}"],
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 0,
        # vLLM >= 0.10.2 exposes this on the OpenAI route. Harmless where unsupported; its ABSENCE
        # from the response is exactly the signal we are testing for.
        "return_token_ids": True,
    }
    try:
        payload = _post(f"{base}/v1/chat/completions", body, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            served_models=served,
            findings=[f"probe failed: HTTP {exc.code}: {detail}"],
        )
    except Exception as exc:  # noqa: BLE001
        return LLMReport(
            ok=False,
            llm_url=base,
            model=model,
            served_models=served,
            findings=[f"probe failed: {type(exc).__name__}: {str(exc)[:300]}"],
        )

    report = check_upstream_response(payload)
    choice = (payload.get("choices") or [{}])[0]
    return LLMReport(
        ok=report.ok,
        llm_url=base,
        model=model,
        served_models=served,
        findings=[str(f) for f in report.findings],
        n_prompt_ids=len(payload.get("prompt_token_ids") or []),
        n_completion_ids=len(choice.get("token_ids") or []),
    )


def require_llm(llm_url: str, model: str, *, timeout: float = 120.0) -> LLMReport:
    """`validate_llm`, but raises instead of returning a failed report.

    For the startup path, where continuing past a bad engine is never the right behaviour.
    """
    report = validate_llm(llm_url, model, timeout=timeout)
    if not report.ok:
        raise RuntimeError(
            report.summary()
            + "\n\nA vLLM server must be started with:"
            + "\n  --return-tokens-as-token-ids --logprobs-mode processed_logprobs"
            + "\nWithout them the engine returns text with no token ids, every rebuilt training row"
            + " is empty, and nothing downstream reports an error."
        )
    return report
