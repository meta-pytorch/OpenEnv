# Dockerfile for Pokemon Battle Environment
# This image provides Pokemon battles via poke-env + Pokemon Showdown
#
# The container runs TWO services:
#   - Pokemon Showdown server (Node.js) on port 8000
#   - OpenEnv HTTP server (FastAPI) on port 9000

# Stage 1: Build Pokemon Showdown
FROM node:18-slim AS showdown-builder

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /pokemon-showdown

RUN git clone https://github.com/smogon/pokemon-showdown.git . && \
    npm install && \
    cp config/config-example.js config/config.js

# Stage 2: Build OpenEnv base (can be overridden for CI/CD)
ARG BASE_IMAGE
FROM ${BASE_IMAGE:-openenv-base:latest} AS final

# Install Node.js for running Pokemon Showdown
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy Pokemon Showdown from builder
COPY --from=showdown-builder /pokemon-showdown /pokemon-showdown

# Install poke-env and dependencies
RUN pip install --no-cache-dir \
    poke-env>=0.9.0 \
    gymnasium>=0.29.0

# Copy OpenEnv core (base image already set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy Pokemon environment code
COPY src/envs/pokemon_env/ /app/src/envs/pokemon_env/

# Copy README for web interface documentation
COPY src/envs/pokemon_env/README.md /app/README.md

# Pokemon environment variables
ENV POKEMON_BATTLE_FORMAT=gen9randombattle
ENV POKEMON_PLAYER_USERNAME=player
ENV POKEMON_REWARD_MODE=sparse
ENV POKEMON_MAX_TURNS=1000

# Expose ports (8000=Showdown, 9980=OpenEnv)
EXPOSE 8000 9980

# Create supervisor config for managing both processes
RUN echo '[supervisord]\n\
nodaemon=true\n\
logfile=/dev/null\n\
logfile_maxbytes=0\n\
\n\
[program:showdown]\n\
command=node pokemon-showdown start --no-security\n\
directory=/pokemon-showdown\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/fd/1\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/fd/2\n\
stderr_logfile_maxbytes=0\n\
startsecs=5\n\
\n\
[program:openenv]\n\
command=uvicorn envs.pokemon_env.server.app:app --host 0.0.0.0 --port 9980\n\
directory=/app\n\
environment=PYTHONPATH="/app/src"\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/dev/fd/1\n\
stdout_logfile_maxbytes=0\n\
stderr_logfile=/dev/fd/2\n\
stderr_logfile_maxbytes=0\n\
startsecs=10\n' > /etc/supervisor/conf.d/pokemon-env.conf

# Health check (check both services)
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000 && curl -f http://localhost:9980/health || exit 1

# Run supervisor to manage both processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
