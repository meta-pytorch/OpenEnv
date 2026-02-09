# Use the base image we built locally
FROM envtorch-base:latest

# 1. Install dependencies
# FIX: Source path is envs/..., not src/envs/...
COPY envs/grid_world_env/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2. Copy Core Utilities
# (This part was correct based on your 'find' command earlier)
COPY src/openenv/core/ /app/src/openenv/core/

# 3. Copy Your Environment
# FIX: Source path must match your repo root (envs/grid_world_env)
COPY envs/grid_world_env/ /app/envs/grid_world_env/

# 4. Environment Variables
# Add /app to PYTHONPATH so python can find 'envs.grid_world_env'
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

# 5. Run the Server
# Note: We run from the module path 'envs.grid_world_env.server.app'
CMD ["uvicorn", "envs.grid_world_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]