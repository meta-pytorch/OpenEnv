# src/envs/finqa_env/server/app.py
"""
FastAPI server for the FinQA environment.

Environment Variables:
    FINQA_DATA_PATH: Path to data directory (default: /app/src/envs/finqa_env/data)
    FINQA_MAX_STEPS: Maximum tool calls per episode (default: 50)
    FINQA_TASK: Task name (default: finqa)
"""

import os

from core.env_server import create_app
from ..models import FinQAAction, FinQAObservation, get_tool_schemas
from .finqa_environment import FinQAEnvironment

DATA_PATH = os.environ.get("FINQA_DATA_PATH", "/app/src/envs/finqa_env/data")
MAX_STEPS = int(os.environ.get("FINQA_MAX_STEPS", "50"))
TASK = os.environ.get("FINQA_TASK", "finqa")

env = FinQAEnvironment(
    data_path=DATA_PATH,
    max_steps=MAX_STEPS,
    task=TASK,
)

app = create_app(env, FinQAAction, FinQAObservation, env_name="finqa_env")


@app.get("/tools")
def get_tools():
    """Return tool schemas in OpenAI function calling format."""
    return get_tool_schemas()
