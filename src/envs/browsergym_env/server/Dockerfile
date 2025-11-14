ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system dependencies for Playwright and browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    fonts-unifont \
    fonts-noto-color-emoji \
    # Additional dependencies
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install BrowserGym and related packages
COPY src/envs/browsergym_env/server/requirements.txt /tmp/browsergym_requirements.txt
RUN pip install --no-cache-dir -r /tmp/browsergym_requirements.txt && \
    rm /tmp/browsergym_requirements.txt

# Install Playwright browsers (Chromium by default)
RUN playwright install chromium

# Install MiniWoB++ tasks
RUN pip install browsergym-miniwob
RUN git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus.git /app/miniwob-plusplus

# Copy OpenEnv core and browsergym_env code
WORKDIR /app
COPY src/core/ /app/src/core/
COPY src/envs/browsergym_env/ /app/src/envs/browsergym_env/
COPY src/envs/browsergym_env/README.md /app/README.md
RUN chmod +x /app/src/envs/browsergym_env/server/start.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME="click-test"
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720
ENV BROWSERGYM_TIMEOUT=10000
ENV MINIWOB_HTML_DIR=/app/miniwob-plusplus/miniwob/html
ENV MINIWOB_HTTP_PORT=8888
ENV MINIWOB_URL=http://127.0.0.1:8888/miniwob/

# For WebArena tasks, these should be set by the user when running the container:
# ENV SHOPPING=
# ENV SHOPPING_ADMIN=
# ENV REDDIT=
# ENV GITLAB=
# ENV MAP=
# ENV WIKIPEDIA=
# ENV HOMEPAGE=

EXPOSE 8000
EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/src/envs/browsergym_env/server/start.sh"]
