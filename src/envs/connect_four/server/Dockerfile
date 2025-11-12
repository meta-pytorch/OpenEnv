FROM python:3.11-slim

# System basics (git not strictly required for OpenSpiel but handy for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Python deps
# - open_spiel from PyPI (>=1.6 ships Linux wheels)
# - pin numpy<2.0 for broad compatibility with older stacks
RUN pip install --no-cache-dir "fastapi>=0.112" "uvicorn[standard]>=0.30" "numpy>=1.24,<2.0" "open_spiel>=1.6"

# Copy project
WORKDIR /app
COPY . /app/

# Defaults (override at runtime)
ENV PORT=8020
ENV OPENSPIEL_GAME=connect_four
ENV CONNECT4_AUTOPLAY_OPPONENT=false
ENV CONNECT4_OPP_POLICY=random

EXPOSE 8020
CMD ["sh", "-c", "uvicorn src.envs.connect_four.server.app:app --host 0.0.0.0 --port ${PORT}"]
