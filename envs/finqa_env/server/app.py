# envs/finqa_env/server/app.py
"""
FastAPI server for the FinQA environment.

Environment Variables:
    FINQA_DATA_PATH: Path to data directory (default: /app/env/data)
    FINQA_MAX_STEPS: Maximum tool calls per episode (default: 50)
    FINQA_TASK: Task name (default: finqa)
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from .finqa_environment import FinQAEnvironment

DATA_PATH = os.environ.get("FINQA_DATA_PATH", "/app/env/data")
MAX_STEPS = int(os.environ.get("FINQA_MAX_STEPS", "50"))
TASK = os.environ.get("FINQA_TASK", "finqa")


def _env_factory():
    """Create a new FinQAEnvironment instance for each session."""
    return FinQAEnvironment(
        data_path=DATA_PATH,
        max_steps=MAX_STEPS,
        task=TASK,
    )


# Pass the class (factory) instead of instance for WebSocket session support
# Use MCP types for action/observation since this is a pure MCP environment
app = create_app(
    _env_factory, CallToolAction, CallToolObservation, env_name="finqa_env"
)
