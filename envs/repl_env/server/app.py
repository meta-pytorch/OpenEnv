# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the REPL Environment.

This module creates an HTTP server that exposes the REPLEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

The server includes llm_query and llm_query_batched support via HuggingFace Inference API,
enabling the Recursive Language Model (RLM) paradigm.

LLM Token Configuration:
    1. Client can pass `hf_token` in reset() - RECOMMENDED
    2. Server fallback: HF_TOKEN environment variable

LLM functions are created dynamically in REPLEnvironment.reset() based on the
available token (client or server).

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server

Environment Variables:
    HF_TOKEN: Fallback HuggingFace API token (client token takes priority)
    LLM_MODEL: Model to use for llm_query/llm_query_batched (default: Qwen/Qwen3-Coder-480B-A35B-Instruct)
"""

import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from ..models import REPLAction, REPLObservation
    from .repl_environment import REPLEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from models import REPLAction, REPLObservation
    from server.repl_environment import REPLEnvironment


# ============== LLM CONFIGURATION ==============
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# ===============================================

# Log LLM configuration
if HF_TOKEN:
    print(f"[REPL Server] LLM support ENABLED (server token configured)")
    print(f"[REPL Server] Default model: {LLM_MODEL}")
else:
    print("[REPL Server] No server HF_TOKEN configured")
    print(
        "[REPL Server] LLM functions will be enabled if client passes hf_token in reset()"
    )

# Simple factory - LLM functions are created dynamically in reset() based on token
env_factory = REPLEnvironment

# Create the app with web interface and README integration
app = create_app(env_factory, REPLAction, REPLObservation, env_name="repl_env")


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.repl_env.server.app
        openenv serve repl_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
