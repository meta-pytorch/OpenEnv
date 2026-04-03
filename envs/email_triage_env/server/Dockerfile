FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn requests pydantic openai \
 && pip install --no-cache-dir -e /app

ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["sh", "-c", "if [ -d /app/server ]; then uvicorn server.app:app --host 0.0.0.0 --port 8000; else uvicorn envs.email_triage_env.server.app:app --host 0.0.0.0 --port 8000; fi"]
