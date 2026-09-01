FROM python:3.12-slim AS builder

WORKDIR /app/env
COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.10.4
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev --no-editable; \
    else \
        uv sync --no-dev --no-editable; \
    fi

FROM python:3.12-slim

WORKDIR /app/env
COPY --from=builder /app/env /app/env

ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENENV_TB_DATA_CACHE=/data/thinkingbox_bench

RUN mkdir -p /data/thinkingbox_bench
VOLUME ["/data/thinkingbox_bench"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "thinkingbox_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
