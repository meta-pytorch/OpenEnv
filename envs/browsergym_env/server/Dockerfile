# Use public Python base image for HuggingFace compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files first (for better caching)
COPY . .

# Make start script executable
RUN chmod +x /app/env/server/start.sh

# Install Python dependencies using pip install -e . (from pyproject.toml)
RUN pip install --no-cache-dir -e .

# Install Playwright browsers (Chromium by default)
# Use python -m since playwright command might not be in PATH
RUN python -m playwright install chromium

# Install MiniWoB++ tasks
RUN git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus.git /app/miniwob-plusplus

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME="click-test"
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720
ENV BROWSERGYM_TIMEOUT=10000
ENV BROWSERGYM_PORT=8000
ENV MINIWOB_HTML_DIR=/app/miniwob-plusplus/miniwob/html
ENV MINIWOB_HTTP_PORT=8888
ENV MINIWOB_URL=http://127.0.0.1:8888/miniwob/
ENV ENABLE_WEB_INTERFACE=true

# For WebArena tasks, these should be set by the user when running the container:
# ENV SHOPPING=
# ENV SHOPPING_ADMIN=
# ENV REDDIT=
# ENV GITLAB=
# ENV MAP=
# ENV WIKIPEDIA=
# ENV HOMEPAGE=

# Expose ports
EXPOSE 8000
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server using the start script
CMD ["/app/env/server/start.sh"]
