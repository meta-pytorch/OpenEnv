FROM python:3.11-slim

WORKDIR /app

# Copy all warehouse environment files (for HF Spaces deployment)
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0

# Expose port
EXPOSE 8000

# Environment variables with defaults
ENV DIFFICULTY_LEVEL=2
ENV GRID_WIDTH=0
ENV GRID_HEIGHT=0
ENV NUM_PACKAGES=0
ENV MAX_STEPS=0
ENV RANDOM_SEED=0

# Set Python path to include current directory
ENV PYTHONPATH=/app

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
