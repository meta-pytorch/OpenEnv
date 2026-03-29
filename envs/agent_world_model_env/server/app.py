"""
Each environment in Agent World Model is a self-contained FastAPI application with SQLAlchemy/SQLite backend and MCP tool interface.

Usage:
    # From repo root:
    PYTHONPATH=src:envs uvicorn envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8000

    # From env directory:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

IMPORTANT:
    AWM environment requires stateful WebSocket connections. HTTP endpoints
    (/reset, /step) are disabled because each HTTP request creates a new
    environment instance, losing subprocess state between calls.
"""

from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import AWMAction, AWMObservation
    from .awm_environment import AWMEnvironment
    from .config import MAX_CONCURRENT_ENVS
    from .data_loader import AWMDataLoader
    from .session_registry import registry as _registry
except ImportError:
    from models import AWMAction, AWMObservation
    from openenv.core.env_server.http_server import create_app
    from server.awm_environment import AWMEnvironment
    from server.config import MAX_CONCURRENT_ENVS
    from server.data_loader import AWMDataLoader
    from server.session_registry import registry as _registry

_shared_data_loader = AWMDataLoader()

app = create_app(
    lambda: AWMEnvironment(data_loader=_shared_data_loader),
    AWMAction,
    AWMObservation,
    env_name="agent_world_model_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)

# Disable HTTP endpoints - AWM requires stateful WebSocket connections
# Remove the default /reset and /step routes, then add error handlers
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

# Remove existing /reset and /step routes
app.routes[:] = [
    r for r in app.routes if getattr(r, "path", None) not in ("/reset", "/step")
]


@app.post("/reset", tags=["disabled"])
async def reset_not_supported():
    """HTTP reset is disabled - use WebSocket /ws endpoint."""
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.post("/step", tags=["disabled"])
async def step_not_supported():
    """HTTP step is disabled - use WebSocket /ws endpoint."""
    return JSONResponse(status_code=400, content=_HTTP_NOT_SUPPORTED_RESPONSE)


@app.get("/stats", tags=["monitoring"])
async def stats():
    """Return session registry stats."""
    return JSONResponse(content=_registry.get_stats())


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
