# Pre-baked image for HFSandboxBackend: opencode + proxy deps installed ahead of
# time so rollouts skip the cold install. Use with OpenCodeConfig(sandbox_home="/root").
#
# Build context = envs/opencode_env:
#   docker build -f sandbox/hf_image/Dockerfile -t <user>/opencode-rl:latest .
#   docker push <user>/opencode-rl:latest

FROM python:3.12

# Proxy deps that interception.py imports.
RUN pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" "httpx>=0.27"

RUN curl -fsSL https://opencode.ai/install | bash \
    && /root/.opencode/bin/opencode --version

# Directory layout the harness expects, plus the pre-copied proxy source.
RUN mkdir -p /root/.config/opencode \
             /root/logs/agent \
             /root/logs/verifier \
             /root/task \
             /root/workdir \
             /root/proxy
COPY interception.py /root/proxy/interception.py

WORKDIR /root/workdir
