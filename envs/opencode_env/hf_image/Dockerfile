# Pre-baked image for HFSandboxBackend: opencode + proxy deps installed ahead of
# time so rollouts skip the cold install. Use with OpenCodeConfig(sandbox_home="/root").
#
# Build context = envs/opencode_env:
#   docker build -f hf_image/Dockerfile -t <user>/opencode-rl:latest .
#   docker push <user>/opencode-rl:latest

FROM python:3.12

# opencode's installer is fetched with curl; the base image doesn't guarantee it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Proxy deps that interception.py imports.
RUN pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" "httpx>=0.27"

RUN curl -fsSL https://opencode.ai/install | bash \
    && /root/.opencode/bin/opencode --version

# Directory layout the harness expects. The proxy source (interception.py) lives in
# openenv.core.sandbox and is uploaded by the harness at rollout start, so it is not
# baked here (matches the pi-sandbox image).
RUN mkdir -p /root/.config/opencode \
             /root/logs/agent \
             /root/logs/verifier \
             /root/task \
             /root/workdir \
             /root/proxy

WORKDIR /root/workdir
