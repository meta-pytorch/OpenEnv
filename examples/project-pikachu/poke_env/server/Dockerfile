# Dockerfile for Pokemon Battle Environment
# This image provides Pokemon battles via poke-env + Pokemon Showdown
#
# The container runs TWO services:
#   - Pokemon Showdown server (Node.js) on port 8000
#   - OpenEnv HTTP server (FastAPI) on port 9980

# Stage 1: Build Pokemon Showdown
FROM node:18-slim AS showdown-builder

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /pokemon-showdown

RUN git clone https://github.com/smogon/pokemon-showdown.git . && \
    npm install && \
    cp config/config-example.js config/config.js

# Stage 2: Build on Python base with FastAPI
FROM python:3.11-slim AS final

# Install Node.js, supervisor, and required tools
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Pokemon Showdown from builder
COPY --from=showdown-builder /pokemon-showdown /pokemon-showdown

# Set working directory
WORKDIR /app

# Install dependencies
COPY src/envs/pokemon_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy Pokemon environment code
COPY src/envs/pokemon_env/ /app/src/envs/pokemon_env/

# Set Python path
ENV PYTHONPATH=/app/src

# Configure supervisor to run both services
RUN mkdir -p /var/log/supervisor && \
    echo '[supervisord]\n\
nodaemon=true\n\
logfile=/var/log/supervisor/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[program:pokemon-showdown]\n\
command=bash -c "find /pokemon-showdown/logs -type f -name \"*.txt\" -delete && find /pokemon-showdown/logs -type f -name \"*.jsonl\" -delete && node pokemon-showdown start --no-security"\n\
directory=/pokemon-showdown\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
startsecs=5\n\
priority=1\n\
\n\
[program:openenv-api]\n\
command=bash -c "sleep 8 && uvicorn envs.pokemon_env.server.app:app --host 0.0.0.0 --port 9980"\n\
directory=/app\n\
environment=PYTHONPATH="/app/src"\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
startsecs=10\n\
priority=2\n' > /etc/supervisor/conf.d/pokemon-env.conf

# Pokemon environment variables
ENV POKEMON_BATTLE_FORMAT=gen8randombattle
ENV POKEMON_PLAYER_USERNAME=player

# Expose both ports (8000=Showdown, 9980=OpenEnv)
EXPOSE 8000 9980

# Health check (check both services)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000 && curl -f http://localhost:9980/health || exit 1

# Run supervisor to manage both processes
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
