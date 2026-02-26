FROM python:3.12-slim

# Disable apt sandboxing for podman --network=host
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /opt/app

# Copy everything and install
COPY pyproject.toml README.md ./
COPY agentkernel/ agentkernel/
RUN uv pip install --system --no-cache ./agentkernel[all]

ENV PYTHONPATH=/opt/app
WORKDIR /workspace
EXPOSE 9000
ENTRYPOINT ["python", "-m", "agentic.kernel.core.runner"]
CMD ["--config", "/workspace/config.json"]
