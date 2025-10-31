ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install any additional dependencies
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    ale-py>=0.8.0 \
    numpy>=1.24.0
# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/connect4_env/ /app/src/envs/connect4_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.connect4_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]