"""FastAPI entrypoint for the Connect4 OpenEnv server."""

from core.env_server.http_server import create_app

from ..models import Connect4Action, Connect4Observation
from .connect4_environment import Connect4Environment

env = Connect4Environment()
app = create_app(env, Connect4Action, Connect4Observation, env_name="connect4_env")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
