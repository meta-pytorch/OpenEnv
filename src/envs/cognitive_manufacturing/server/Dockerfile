# Dockerfile for Cognitive Manufacturing Environment
# This creates a containerized version of the environment server

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy OpenEnv (assuming it's in parent directory)
COPY OpenEnv/ /app/OpenEnv/
RUN pip install --no-cache-dir -e /app/OpenEnv

# Copy our environment code
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose the server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the server
CMD ["python", "-m", "uvicorn", "envs.cognitive_manufacturing.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
