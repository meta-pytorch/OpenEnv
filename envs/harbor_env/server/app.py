"""ASGI entry point for a deployed harbor_env.

Everything is read from the environment so the same image serves any dataset and any engine without
a rebuild — which is what makes this deployable to a Space:

    OPENENV_DATASETS     comma-separated dataset specs (HF repo id, local dir, harbor name@version)
    OPENENV_LLM_URL   OpenAI-spec inference endpoint
    OPENENV_MODEL        served model id; read from the engine when it serves exactly one
    E2B_API_KEY / MODAL_TOKEN_ID+MODAL_TOKEN_SECRET   whichever sandboxes you want offered

The capture proxy rides on this same app rather than on a second port. A Space publishes exactly one
port and one URL, so the proxy is mounted at `/capture` and the sandbox reaches it at
`https://<space>.hf.space/capture`. Nothing is forwarded and no second listener is opened.
"""

from __future__ import annotations

import os

from openenv.harbor.serving import HarborService, build_app

_DATASETS = [
    d.strip() for d in os.environ.get("OPENENV_DATASETS", "").split(",") if d.strip()
]
_LLM_URL = os.environ.get("OPENENV_LLM_URL", "")
_MODEL = os.environ.get("OPENENV_MODEL", "")
_LLM: dict = {}

# Ask the endpoint what it serves when `OPENENV_MODEL` was not set, the same way `harbor serve` does.
# Without this the proxy has no served model id and stops rewriting `model` on the way upstream, so
# whatever name the harness happened to use is forwarded verbatim and the engine rejects it. The
# report is kept so `capabilities()` can state whether capture is actually supported here.
if _LLM_URL:
    try:
        from openenv.core.harness.capture.validate_llm import list_models, validate_llm

        if not _MODEL:
            served = list_models(_LLM_URL)
            _MODEL = served[0] if len(served) == 1 else ""
        if _MODEL:
            _report = validate_llm(_LLM_URL, _MODEL)
            _LLM = {
                "url": _LLM_URL,
                "model": _report.model,
                "ok": _report.ok,
                "findings": _report.findings,
                "served_models": _report.served_models,
            }
    except Exception as exc:  # noqa: BLE001 - a Space must still boot so the UI can show the fault
        _LLM = {
            "url": _LLM_URL,
            "model": _MODEL,
            "ok": False,
            "findings": [
                f"could not reach the LLM at startup: {type(exc).__name__}: {exc}"
            ],
        }

# Resolve capture before the app is built. A Space gives no separate boot hook, the UI needs the
# proxy's public URL to exist by the time anyone presses Run, and `build_app` has to see the service
# in order to mount it.
if _LLM_URL:
    _service = HarborService(
        llm_url=_LLM_URL,
        model=_MODEL,
        datasets=_DATASETS,
        capture_port=int(os.environ.get("OPENENV_CAPTURE_PORT", "8100")),
        expose=os.environ.get("OPENENV_EXPOSE", "gradio"),
    )
    # On a Space this only computes the public URL and flags the app for mounting; off one it
    # publishes the capture port the usual way.
    _service.start()
    HarborService.set_current(_service)

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

app = build_app(datasets=_DATASETS, llm_url=_LLM_URL, model=_MODEL, llm=_LLM)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
