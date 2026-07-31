# Pre-baked image for HFSandboxBackend: Node + Claude Code + proxy deps installed
# ahead of time so rollouts skip the cold install. Use with
# ClaudeCodeConfig(sandbox_home="/root").
#
# The interception proxy (openenv.core.sandbox) and the Anthropic shim live
# outside this build context, so they are uploaded by the factory at runtime
# rather than baked here. Only their Python deps are baked.
#
# Build context = envs/claude_code_env:
#   docker build -f hf_image/Dockerfile -t <user>/claude-code-rl:latest .
#   docker push <user>/claude-code-rl:latest

FROM python:3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Proxy and shim deps that interception.py / anthropic_shim.py import.
RUN pip install --no-cache-dir "fastapi>=0.104" "uvicorn[standard]>=0.24" "httpx>=0.27"

# Node 22 (Claude Code requires >=22.19) + the Claude Code CLI, at the paths
# claude_code_runtime expects.
RUN A=$(uname -m); case $A in x86_64) A=x64;; aarch64|arm64) A=arm64;; esac \
    && curl -fsSL "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-$A.tar.xz" | tar -xJ -C /root \
    && ln -sfn "/root/node-v22.19.0-linux-$A" /root/.node \
    && /root/.node/bin/npm install -g --prefix /root/.claude-npm @anthropic-ai/claude-code \
    && /root/.claude-npm/bin/claude --version

# Directory layout the harness expects.
RUN mkdir -p /root/.claude \
             /root/logs/agent \
             /root/logs/verifier \
             /root/task \
             /root/workdir \
             /root/proxy \
             /root/shim

WORKDIR /root/workdir
