# Start from a public, official Python image
FROM python:3.11-slim

# Install system dependencies like curl (for the health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies. This is done in a separate
# step to leverage Docker's layer caching. Dependencies are only re-installed
# when the requirements.txt file changes.
COPY src/envs/dipg_safety_env/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory and PYTHONPATH inside the container
WORKDIR /app
ENV PYTHONPATH="/app/src"

# Copy all the application source code into the container
COPY src/core/ /app/src/core/
COPY src/envs/dipg_safety_env/ /app/src/envs/dipg_safety_env/

# Expose the port the server will run on
EXPOSE 8000

# Add a robust health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# Note: The DIPG_DATASET_PATH must be provided when running this container.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "envs.dipg_safety_env.server.app:app"]
