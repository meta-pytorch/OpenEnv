"""FastAPI application for the Connect4 Environment."""

from openenv.core.env_server import create_app
from ..models import Connect4Action, Connect4Observation
from .connect4_environment import Connect4Environment

# Create the FastAPI app
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(Connect4Environment, Connect4Action, Connect4Observation, env_name="connect4_env")

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)