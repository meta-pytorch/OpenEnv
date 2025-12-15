# FinQA Environment Dockerfile
#
# Build from repository root:
#   docker build -t finqa-env:latest -f src/envs/finqa_env/server/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 finqa-env:latest

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY src/envs/finqa_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install core OpenEnv dependencies (fastapi, uvicorn, etc.)
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    requests>=2.25.0 \
    gunicorn>=21.0.0

# Set PYTHONPATH
ENV PYTHONPATH="/app/src"

# Copy core library
COPY src/core/ /app/src/core/

# Copy environment code
COPY src/envs/finqa_env/ /app/src/envs/finqa_env/

# Environment variables with defaults
ENV FINQA_DATA_PATH="/app/src/envs/finqa_env/data"
ENV FINQA_MAX_STEPS="20"
ENV FINQA_TASK="finqa"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "envs.finqa_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
