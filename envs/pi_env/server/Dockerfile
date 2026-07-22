# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# pi_env Dockerfile — mirrors the standard OpenEnv multi-stage uv build used
# by echo_env / repl_env / opencode_env.
#
# Build:
#   docker build -t pi-env .
#
# Run:
#   docker run -p 8000:8000 -e HF_TOKEN=hf_... pi-env

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

ARG BUILD_MODE=in-repo

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# The shared sandbox backend + interception proxy live in opencode_env (kept a
# peer dep in pyproject, not a hard dep, so local dev / training installs stay
# lean). Install it explicitly into the venv so the server can import
# ``opencode_env.sandbox``. TEMPORARY: repoint to the branch carrying
# HFSandboxBackend until it lands in main; removed once the shared sandbox
# consolidates into openenv.core.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
    "openenv-opencode-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/opencode_env"

# ── runtime stage ────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
