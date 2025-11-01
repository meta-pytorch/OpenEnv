from core.env_server import create_fastapi_app
from ..models import Connect4Action, Connect4Observation
from .connect4_environment import Connect4Environment

env = Connect4Environment()
app = create_fastapi_app(env, Connect4Action, Connect4Observation)

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)