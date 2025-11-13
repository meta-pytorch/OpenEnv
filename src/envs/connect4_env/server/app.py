from openenv_core.env_server.http_server import create_app

from ..models import Connect4Action, Connect4Observation
from .connect4_environment import Connect4Environment

env = Connect4Environment()
app = create_app(
    env,
    Connect4Action,
    Connect4Observation,
    env_name="connect4_env",
)


def main(port: int = 8000):
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.port)
