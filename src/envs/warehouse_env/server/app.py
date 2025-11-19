"""
FastAPI server for Warehouse Optimization Environment.

This module creates the HTTP server that exposes the warehouse environment
via REST API endpoints.
"""

import os

from core.env_server import create_app
from envs.warehouse_env.models import WarehouseAction, WarehouseObservation
from envs.warehouse_env.server.warehouse_environment import WarehouseEnvironment
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse


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


# Create FastAPI app using OpenEnv's helper (with web interface if enabled)
app = create_app(warehouse_env, WarehouseAction, WarehouseObservation, env_name="warehouse_env")


# Add custom render endpoints
@app.post("/set-difficulty")
async def set_difficulty(request: dict):
    """Change the difficulty level and reset the environment."""
    try:
        difficulty = int(request.get("difficulty", 2))
        if difficulty < 1 or difficulty > 5:
            return JSONResponse(
                status_code=400,
                content={"error": "Difficulty must be between 1 and 5"}
            )

        # Recreate the warehouse environment with new difficulty
        global warehouse_env
        warehouse_env = WarehouseEnvironment(
            difficulty_level=difficulty,
            grid_width=None,
            grid_height=None,
            num_packages=None,
            max_steps=None,
            random_seed=None,
        )

        # Reset the environment
        observation = warehouse_env.reset()

        return JSONResponse(content={
            "success": True,
            "difficulty": difficulty,
            "grid_size": (warehouse_env.grid_width, warehouse_env.grid_height),
            "num_packages": warehouse_env.num_packages,
            "max_steps": warehouse_env.max_steps,
            "observation": {
                "step_count": observation.step_count,
                "packages_delivered": observation.packages_delivered,
                "total_packages": observation.total_packages,
                "robot_position": observation.robot_position,
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to set difficulty: {str(e)}"}
        )


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

@app.get("/render/html")
async def render_html():
    """Get HTML visualization of warehouse state."""
    try:
        html_content = warehouse_env.render_html()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to render HTML: {str(e)}"}
        )

@app.post("/auto-step")
async def auto_step():
    """Execute one step using a greedy agent."""
    try:
        # Get current observation
        if warehouse_env.is_done:
            return JSONResponse(content={
                "done": True,
                "message": "Episode finished. Reset to start a new episode."
            })

        # Simple greedy policy
        action_id = _get_greedy_action()
        action = WarehouseAction(action_id=action_id)

        # Execute step
        result = warehouse_env.step(action)

        return JSONResponse(content={
            "action": action.action_name,
            "message": result.message,
            "reward": result.reward,
            "done": result.done,
            "step_count": result.step_count,
            "packages_delivered": result.packages_delivered,
            "robot_position": result.robot_position,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to execute auto-step: {str(e)}"}
        )

def _get_greedy_action() -> int:
    """Simple greedy policy with obstacle avoidance."""
    robot_x, robot_y = warehouse_env.robot_position

    # Determine target location
    if warehouse_env.robot_carrying is None:
        # Not carrying: move toward nearest waiting package
        target = None
        min_dist = float('inf')

        for package in warehouse_env.packages:
            if package.status == "waiting":
                px, py = package.pickup_location
                dist = abs(robot_x - px) + abs(robot_y - py)
                if dist < min_dist:
                    min_dist = dist
                    target = (px, py)

        if target is None:
            return 4  # Try to pick up if at location

        target_x, target_y = target
    else:
        # Carrying: move toward dropoff zone
        package = next((p for p in warehouse_env.packages if p.id == warehouse_env.robot_carrying), None)
        if package:
            target_x, target_y = package.dropoff_location
        else:
            return 5  # Try to drop off

    # Check if at target location
    if robot_x == target_x and robot_y == target_y:
        return 4 if warehouse_env.robot_carrying is None else 5

    # Try to move toward target, checking for obstacles
    # Priority: move on axis with larger distance first
    dx = target_x - robot_x
    dy = target_y - robot_y

    # List of possible moves in order of preference
    moves = []

    if abs(dx) > abs(dy):
        # Prioritize horizontal movement
        if dx > 0:
            moves.append((3, robot_x + 1, robot_y))  # RIGHT
        elif dx < 0:
            moves.append((2, robot_x - 1, robot_y))  # LEFT

        if dy > 0:
            moves.append((1, robot_x, robot_y + 1))  # DOWN
        elif dy < 0:
            moves.append((0, robot_x, robot_y - 1))  # UP
    else:
        # Prioritize vertical movement
        if dy > 0:
            moves.append((1, robot_x, robot_y + 1))  # DOWN
        elif dy < 0:
            moves.append((0, robot_x, robot_y - 1))  # UP

        if dx > 0:
            moves.append((3, robot_x + 1, robot_y))  # RIGHT
        elif dx < 0:
            moves.append((2, robot_x - 1, robot_y))  # LEFT

    # Add perpendicular moves as fallback
    if dx == 0 and dy != 0:
        moves.append((3, robot_x + 1, robot_y))  # RIGHT
        moves.append((2, robot_x - 1, robot_y))  # LEFT
    elif dy == 0 and dx != 0:
        moves.append((1, robot_x, robot_y + 1))  # DOWN
        moves.append((0, robot_x, robot_y - 1))  # UP

    # Try moves in order until we find a valid one
    WALL = 1
    SHELF = 2

    for action_id, new_x, new_y in moves:
        # Check bounds
        if 0 <= new_x < warehouse_env.grid_width and 0 <= new_y < warehouse_env.grid_height:
            # Check if cell is passable
            if warehouse_env.grid[new_y][new_x] not in [WALL, SHELF]:
                return action_id

    # If no valid move toward target, try any valid move
    for action_id, dx, dy in [(0, 0, -1), (1, 0, 1), (2, -1, 0), (3, 1, 0)]:
        new_x, new_y = robot_x + dx, robot_y + dy
        if 0 <= new_x < warehouse_env.grid_width and 0 <= new_y < warehouse_env.grid_height:
            if warehouse_env.grid[new_y][new_x] not in [WALL, SHELF]:
                return action_id

    # Last resort: try pickup/dropoff
    return 4 if warehouse_env.robot_carrying is None else 5


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


@app.get("/demo")
async def demo():
    """Serve the interactive demo page."""
    import pathlib
    demo_path = pathlib.Path(__file__).parent / "demo.html"
    if demo_path.exists():
        return FileResponse(demo_path)
    else:
        return HTMLResponse(content="<h1>Demo page not found</h1><p>Please check the server configuration.</p>")


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
