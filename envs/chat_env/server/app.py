# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Chat Environment.

This module creates an HTTP server that exposes the ChatEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Note: This server requires a tokenizer to be initialized. The tokenizer
must be specified when starting the server.

Usage:
    # Development (with auto-reload):
    uvicorn envs.chat_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.chat_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.chat_env.server.app
"""

import os

from openenv.core.env_server import create_app
from openenv.core.env_server.web_interface import create_web_interface_app

from ..models import ChatAction, ChatObservation
from .chat_environment import ChatEnvironment


# Initialize tokenizer based on environment variable
def get_tokenizer():
    """Get tokenizer from environment or use a mock for testing."""
    tokenizer_name = os.environ.get("TOKENIZER_NAME", "gpt2")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Loaded tokenizer: {tokenizer_name}")
        return tokenizer
    except ImportError:
        print(
            "Warning: transformers not installed, using mock tokenizer for testing only"
        )
        # Use mock tokenizer from tests
        import sys
        from pathlib import Path

        # Add parent directory to path to import test utilities
        test_path = Path(__file__).parent
        sys.path.insert(0, str(test_path))

        from test_chat_env import MockTokenizer

        return MockTokenizer()


# Get system prompt from environment
system_prompt = os.environ.get("SYSTEM_PROMPT", None)

# Create the environment instance with tokenizer
tokenizer = get_tokenizer()
env = ChatEnvironment(tokenizer=tokenizer, system_prompt=system_prompt)

# Create the FastAPI app with web interface and README integration
app = create_app(env, ChatAction, ChatObservation, env_name="chat_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
