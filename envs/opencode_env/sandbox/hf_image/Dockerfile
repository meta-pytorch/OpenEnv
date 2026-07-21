# Pre-baked image for the OpenCode agent on HF Sandbox (huggingface_hub.Sandbox).
#
# Mirror of the E2B template in ../build_template.py: bakes opencode + the
# in-sandbox proxy deps so per-rollout cold-install (~3 min) drops to ~30s.
# Runtime install also works WITHOUT this image (OpenCodeSessionFactory bootstrap
# + verified HF-sandbox egress), so this is a speed optimization, not a requirement.
#
# HF sandboxes exec as ROOT ($HOME=/root, verified), so everything lives under
# /root and you must pair it with OpenCodeConfig(sandbox_home="/root").
#
# Build context = envs/opencode_env (so `interception.py` is COPY-able):
#   docker build -f sandbox/hf_image/Dockerfile -t <user>/opencode-rl:latest .
#   docker push <user>/opencode-rl:latest
# Then: HFSandboxBackend(image="<user>/opencode-rl:latest")
#       + OpenCodeConfig(sandbox_home="/root")

FROM python:3.12

# Proxy deps (fastapi/uvicorn/httpx) — what interception.py imports.
RUN pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" "httpx>=0.27"

# opencode CLI -> /root/.opencode/bin/opencode (root exec user).
RUN curl -fsSL https://opencode.ai/install | bash \
    && /root/.opencode/bin/opencode --version

# Directory layout expected by the harness + the pre-copied proxy source.
RUN mkdir -p /root/.config/opencode \
             /root/logs/agent \
             /root/logs/verifier \
             /root/task \
             /root/workdir \
             /root/proxy
COPY interception.py /root/proxy/interception.py

WORKDIR /root/workdir
