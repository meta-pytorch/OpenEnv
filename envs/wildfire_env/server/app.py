# server/app.py
import os
from fastapi.responses import HTMLResponse
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import asdict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server import create_fastapi_app
    from openenv.core.env_server.web_interface import load_environment_metadata, WebInterfaceManager
    from openenv.core.env_server.types import Action, Observation
    from ..models import WildfireAction, WildfireObservation
    from .wildfire_environment import WildfireEnvironment
    from .wildfire_web_interface import get_wildfire_web_interface_html
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.env_server import create_fastapi_app
    from openenv_core.env_server.web_interface import load_environment_metadata, WebInterfaceManager
    from openenv_core.env_server.types import Action, Observation
    from wildfire_env.models import WildfireAction, WildfireObservation
    from wildfire_env.server.wildfire_environment import WildfireEnvironment
    from wildfire_env.server.wildfire_web_interface import get_wildfire_web_interface_html

W = int(os.getenv("WILDFIRE_WIDTH", "16"))
H = int(os.getenv("WILDFIRE_HEIGHT", "16"))

# Factory function to create WildfireEnvironment instances
def create_wildfire_environment():
    """Factory function that creates WildfireEnvironment with config."""
    return WildfireEnvironment(width=W, height=H)

# Check if web interface should be enabled
# This can be controlled via environment variable
enable_web = (
    os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
)

if enable_web:
    # Create an instance for metadata loading (load_environment_metadata needs an instance)
    env_instance = create_wildfire_environment()
    metadata = load_environment_metadata(env_instance, 'wildfire_env')

    # Create base app without web interface first
    # Pass the factory function instead of an instance for WebSocket session support
    app = create_fastapi_app(create_wildfire_environment, WildfireAction, WildfireObservation)

    # Create web interface manager (needed for /web/reset, /web/step, /ws endpoints)
    # WebInterfaceManager expects an Environment instance, not a callable
    web_manager = WebInterfaceManager(env_instance, WildfireAction, WildfireObservation, metadata)

    # Add our custom wildfire interface route
    @app.get("/web", response_class=HTMLResponse)
    async def wildfire_web_interface():
        """Custom wildfire-specific web interface."""
        return get_wildfire_web_interface_html(metadata)

    # Add web interface endpoints (these are needed for the interface to work)
    @app.get("/web/metadata")
    async def web_metadata():
        """Get environment metadata."""
        return asdict(metadata)

    @app.websocket("/ws/ui")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await web_manager.connect_websocket(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            await web_manager.disconnect_websocket(websocket)

    @app.post("/web/reset")
    async def web_reset():
        """Reset endpoint for web interface."""
        return await web_manager.reset_environment()

    @app.post("/web/step")
    async def web_step(request: dict):
        """Step endpoint for web interface."""
        action_data = request.get("action", {})
        return await web_manager.step_environment(action_data)

    @app.get("/web/state")
    async def web_state():
        """State endpoint for web interface."""
        return web_manager.get_state()


def main():
    """Main entry point for running the server."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
