# CARLA Environment - Real Mode with Full CARLA Server (Standalone)
# Complete self-contained deployment with CARLA 0.10.0 server
#
# Requirements:
#   - GPU: NVIDIA T4 (minimum) or A10G (recommended)
#   - RAM: 16GB+
#   - Disk: ~15GB
#   - HuggingFace Space Hardware: GPU T4 or better
#
# Cost on HuggingFace Spaces:
#   - T4 GPU: ~$0.60/hour (~$432/month if running 24/7)
#   - A10G GPU: ~$1.10/hour (~$792/month if running 24/7)
#
# Build time: 30-60 minutes (downloads ~10GB of CARLA)

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # CARLA dependencies
    wget \
    curl \
    ca-certificates \
    git \
    libomp5 \
    libpng16-16 \
    libvulkan1 \
    # For offscreen rendering
    xvfb \
    libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /opt

# Download and extract CARLA 0.10.0 (UE5.5, released Dec 2024)
# Official release: https://github.com/carla-simulator/carla/releases/tag/0.10.0
RUN wget -q https://tiny.carla.org/carla-0-10-0-linux-tar -O /tmp/CARLA_0.10.0.tar.gz \
    && mkdir -p /opt/carla_temp \
    && tar -xzf /tmp/CARLA_0.10.0.tar.gz -C /opt/carla_temp \
    && rm /tmp/CARLA_0.10.0.tar.gz \
    # Handle both flat and nested extraction structures
    && if [ -f /opt/carla_temp/CarlaUE4.sh ]; then \
         mv /opt/carla_temp /opt/carla; \
       else \
         mv /opt/carla_temp/* /opt/carla || mv /opt/carla_temp/CARLA_* /opt/carla; \
         rmdir /opt/carla_temp; \
       fi \
    && chmod +x /opt/carla/CarlaUE4.sh

# Create symbolic link for CARLA
RUN ln -s /opt/carla/CarlaUE4.sh /usr/local/bin/CarlaUE4

# Set CARLA environment variables
ENV CARLA_ROOT=/opt/carla
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install OpenEnv core from GitHub
RUN pip install --no-cache-dir git+https://github.com/meta-pytorch/OpenEnv.git

# Copy and install environment dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Install CARLA Python client (UE5 API, compatible with CARLA 0.9.x and 0.10.x)
RUN pip install --no-cache-dir carla-ue5-api==0.10.0

# Copy CARLA environment code
COPY . /app/carla_env/

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH}

# Environment variables for REAL mode
ENV CARLA_MODE=real
ENV CARLA_SCENARIO=trolley_saves
ENV CARLA_HOST=localhost
ENV CARLA_PORT=2000

# CARLA server settings for offscreen rendering
ENV SDL_VIDEODRIVER=offscreen
ENV DISPLAY=

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
EXPOSE 2000

# Startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚗 Starting CARLA Server 0.10.0 (UE5.5) in offscreen mode..."\n\
cd /opt/carla\n\
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 -fps=20 &\n\
CARLA_PID=$!\n\
\n\
echo "⏳ Waiting for CARLA server to be ready (30 seconds)..."\n\
sleep 30\n\
\n\
echo "🔌 Testing CARLA connection..."\n\
python3 -c "import carla; client = carla.Client('\''localhost'\'', 2000); client.set_timeout(10.0); print('\''✅ CARLA server is ready!'\'')" || {\n\
    echo "❌ CARLA server failed to start"\n\
    kill $CARLA_PID 2>/dev/null || true\n\
    exit 1\n\
}\n\
\n\
echo "🚀 Starting OpenEnv FastAPI server..."\n\
cd /app\n\
uvicorn carla_env.server.app:app --host 0.0.0.0 --port 8000\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
