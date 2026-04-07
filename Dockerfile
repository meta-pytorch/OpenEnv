FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY . /app

# Reuse dependencies bundled in openenv-base to reduce build-time network failures.
ENV PYTHONPATH=/app/src:/app
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uvicorn", "envs.email_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
