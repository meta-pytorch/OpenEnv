"""
FastAPI server for Warehouse Optimization Environment.

This module creates the HTTP server that exposes the warehouse environment
via REST API endpoints.
"""

import os

from core.env_server import create_fastapi_app
from envs.warehouse_env.models import WarehouseAction, WarehouseObservation
from envs.warehouse_env.server.warehouse_environment import WarehouseEnvironment
from fastapi import FastAPI
from fastapi.responses import JSONResponse


# Get configuration from environment variables
DIFFICULTY_LEVEL = int(os.getenv("DIFFICULTY_LEVEL", "2"))
GRID_WIDTH = int(os.getenv("GRID_WIDTH", "0")) or None
GRID_HEIGHT = int(os.getenv("GRID_HEIGHT", "0")) or None
NUM_PACKAGES = int(os.getenv("NUM_PACKAGES", "0")) or None
MAX_STEPS = int(os.getenv("MAX_STEPS", "0")) or None
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "0")) or None


# Create the warehouse environment instance
warehouse_env = WarehouseEnvironment(
    difficulty_level=DIFFICULTY_LEVEL,
    grid_width=GRID_WIDTH,
    grid_height=GRID_HEIGHT,
    num_packages=NUM_PACKAGES,
    max_steps=MAX_STEPS,
    random_seed=RANDOM_SEED,
)


# Create FastAPI app using OpenEnv's helper
app = create_fastapi_app(warehouse_env, WarehouseAction, WarehouseObservation)


# Add custom render endpoint
@app.get("/render")
async def render():
    """Get ASCII visualization of warehouse state."""
    try:
        ascii_art = warehouse_env.render_ascii()
        return JSONResponse(content={"ascii": ascii_art})
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to render: {str(e)}"}
        )


# Add health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": "warehouse_env",
        "difficulty_level": DIFFICULTY_LEVEL,
        "grid_size": (warehouse_env.grid_width, warehouse_env.grid_height),
        "num_packages": warehouse_env.num_packages,
        "max_steps": warehouse_env.max_steps,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    """Entry point for warehouse-server command."""
    import uvicorn
    import os
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
