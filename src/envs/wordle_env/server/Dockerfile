# Accept base image as build argument for CI/CD flexibility
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# No additional dependencies needed for Wordle environment
# The base image already contains all required packages

# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/wordle_env/ /app/src/envs/wordle_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.wordle_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
