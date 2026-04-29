"""
Each environment in Agent World Model is a self-contained FastAPI application
with SQLAlchemy/SQLite backend and MCP tool interface.

Usage:
    PYTHONPATH=src:envs uvicorn envs.agent_world_model_env.server.app:app \\
        --host 0.0.0.0 --port 8000

HTTP /reset and /step are disabled because AWM requires stateful WebSocket
connections — each HTTP request would create a fresh environment, dropping
the subprocess and tool cache.
"""

import inspect
import os
import uvicorn

import gradio as gr
from fastapi.responses import JSONResponse, RedirectResponse

from openenv.core.env_server.http_server import create_app

from ..models import AWMAction, AWMObservation
from .awm_environment import AWMEnvironment
from .config import MAX_CONCURRENT_ENVS
from .data_loader import AWMDataLoader
from .session_registry import registry as _registry
from .web_ui import build_awm_gradio_app


_shared_data_loader = AWMDataLoader()


def _env_factory():
    return AWMEnvironment(data_loader=_shared_data_loader)


# openenv-core gained ``gradio_builder`` in 0.2.3; PyPI's 0.2.1 lacks it.
# Pass it through when supported, fall back to a manual swap below otherwise.
_supports_gradio_builder = "gradio_builder" in inspect.signature(create_app).parameters
_create_app_kwargs = (
    {"gradio_builder": build_awm_gradio_app} if _supports_gradio_builder else {}
)

app = create_app(
    _env_factory,
    AWMAction,
    AWMObservation,
    env_name="agent_world_model_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
    **_create_app_kwargs,
)


def _swap_in_custom_gradio_ui() -> None:
    """Replace openenv-core 0.2.1's default Playground UI with the AWM UI.

    Pulls the WebInterfaceManager out of the existing route closures, drops
    the default Gradio mount and the legacy HTTP endpoints (`/web`,
    `/web/reset`, `/web/step`, ...), then mounts our blocks at `/web`.
    """
    if _supports_gradio_builder:
        return
    if os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() not in (
        "true",
        "1",
        "yes",
    ):
        return

    web_manager = None
    metadata = None
    for r in app.routes:
        for cell in getattr(getattr(r, "endpoint", None), "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if web_manager is None and v.__class__.__name__ == "WebInterfaceManager":
                web_manager = v
            if metadata is None and v.__class__.__name__ == "EnvironmentMetadata":
                metadata = v
        if web_manager is not None and metadata is not None:
            break
    if web_manager is None:
        return

    # /web in 0.2.1 is the legacy "HumanAgent Interface" HTMLResponse, not a
    # redirect — drop it together with the rest of the default UI's HTTP API.
    legacy_paths = {
        "/web",
        "/web/reset",
        "/web/step",
        "/web/state",
        "/web/metadata",
        "/ws/ui",
    }
    app.routes[:] = [
        r
        for r in app.routes
        if not (
            (getattr(r, "path", None) == "/web" and r.__class__.__name__ == "Mount")
            or getattr(r, "path", None) in legacy_paths
        )
    ]

    blocks = build_awm_gradio_app(
        web_manager,
        action_fields=None,
        metadata=metadata,
        is_chat_env=False,
        title="agent_world_model_env",
        quick_start_md=None,
    )
    gr.mount_gradio_app(app, blocks, path="/web")


_swap_in_custom_gradio_ui()


_HTTP_NOT_SUPPORTED_RESPONSE = {
    "error": "HTTP mode not supported for AWM environment",
    "reason": "AWM launches subprocesses on reset() that must persist across step() calls. "
    "HTTP is stateless - each request creates a new environment instance, "
    "losing the subprocess and all loaded tools.",
    "solution": "Use WebSocket endpoint instead",
    "examples": [
        "Python: AWMEnv(base_url='http://host:port')  # uses /ws internally",
        "Direct: connect to ws://host:port/ws",
    ],
}

app.routes[:] = [
    r for r in app.routes if getattr(r, "path", None) not in ("/reset", "/step")
]


@app.post("/reset", tags=["disabled"])
async def reset_not_supported():
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.post("/step", tags=["disabled"])
async def step_not_supported():
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.get("/stats", tags=["monitoring"])
async def stats():
    return JSONResponse(content=_registry.get_stats())


def _has_route(path: str) -> bool:
    return any(getattr(r, "path", None) == path for r in app.routes)


# 0.2.1 doesn't auto-redirect / and /web to /web/. HF Spaces hits both.
if not _has_route("/"):

    @app.get("/", include_in_schema=False)
    async def _root_redirect():
        return RedirectResponse(url="/web/")


if not _has_route("/web"):

    @app.get("/web", include_in_schema=False)
    async def _web_redirect():
        return RedirectResponse(url="/web/")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
