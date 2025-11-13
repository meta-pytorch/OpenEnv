# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS runtime

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    ENABLE_WEB_INTERFACE=true

CMD ["python", "-m", "uvicorn", "connect4_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]