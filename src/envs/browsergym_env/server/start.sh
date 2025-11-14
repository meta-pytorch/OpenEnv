#!/usr/bin/env bash
set -euo pipefail

MINIWOB_HTML_DIR=${MINIWOB_HTML_DIR:-/app/miniwob-plusplus/miniwob/html}
MINIWOB_HTTP_PORT=${MINIWOB_HTTP_PORT:-8888}
BROWSERGYM_PORT=${BROWSERGYM_PORT:-8000}

if [ ! -d "${MINIWOB_HTML_DIR}" ]; then
    echo "MiniWoB HTML directory not found at ${MINIWOB_HTML_DIR}" >&2
    exit 1
fi

python -m http.server "${MINIWOB_HTTP_PORT}" --bind 0.0.0.0 --directory "${MINIWOB_HTML_DIR}" &
HTTP_SERVER_PID=$!

sleep 1
if ! kill -0 "${HTTP_SERVER_PID}" 2>/dev/null; then
    echo "Failed to start MiniWoB static server on port ${MINIWOB_HTTP_PORT}" >&2
    exit 1
fi

cleanup() {
    kill "${HTTP_SERVER_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

exec uvicorn envs.browsergym_env.server.app:app --host 0.0.0.0 --port "${BROWSERGYM_PORT}"

