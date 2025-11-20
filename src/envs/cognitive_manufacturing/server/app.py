"""FastAPI server application for cognitive manufacturing environment.

This module creates the HTTP server that exposes the environment via REST API,
following the OpenEnv specification.
"""

from core.env_server import create_fastapi_app
from .environment import CognitiveManufacturingEnvironment
from ..models import ManufacturingAction, ManufacturingObservation


# Create environment instance
env = CognitiveManufacturingEnvironment()

# Create the FastAPI application
app = create_fastapi_app(env, ManufacturingAction, ManufacturingObservation)


def main():
    """Entry point for running the server directly."""
    import uvicorn

    uvicorn.run(
        "envs.cognitive_manufacturing.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
