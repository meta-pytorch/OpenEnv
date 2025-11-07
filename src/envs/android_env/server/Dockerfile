# Android Environment for OpenEnv
# Build with: docker build -t android-env:latest -f src/envs/android_env/server/Dockerfile .
#
# This Dockerfile creates a container with:
# - Android SDK and command-line tools
# - Android Emulator
# - android_env Python package
# - OpenEnv wrapper for android_env
#
# The container requires:
# - KVM access for hardware acceleration (Linux hosts)
# - Significant resources (4GB+ RAM, 4+ CPU cores)
#
# Environment Variables Required:
# - ANDROID_AVD_NAME: Name of the Android Virtual Device
# - ANDROID_TASK_PATH: Path to the task textproto file
#
# Example build:
#   docker build -t android-env:latest -f src/envs/android_env/server/Dockerfile .
#
# Example run:
#   docker run -p 8000:8000 \
#     -e ANDROID_AVD_NAME=Pixel_6_API_33 \
#     -e ANDROID_TASK_PATH=/workspace/tasks/my_task.textproto \
#     -v /path/to/tasks:/workspace/tasks \
#     --device /dev/kvm \
#     android-env:latest

# Accept base image as build argument
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system dependencies for Android SDK and emulator
RUN apt-get update && apt-get install -y \
    # Android SDK dependencies
    wget \
    unzip \
    openjdk-11-jdk \
    # Emulator dependencies
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    xvfb \
    libxkbcommon-x11-0 \
    libpulse0 \
    libxcomposite1 \
    libxcursor1 \
    # Build tools
    build-essential \
    # Hardware acceleration
    qemu-kvm \
    libvirt-daemon-system \
    libvirt-clients \
    bridge-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables for Android
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV ANDROID_AVD_HOME=/root/.android/avd
ENV ANDROID_HOME=${ANDROID_SDK_ROOT}
ENV PATH=${PATH}:${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${ANDROID_SDK_ROOT}/platform-tools:${ANDROID_SDK_ROOT}/emulator

# Create SDK directory
RUN mkdir -p ${ANDROID_SDK_ROOT}

# Download and install Android command-line tools
# Using commandlinetools version 11076708 (latest as of 2024)
WORKDIR /tmp
RUN wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip && \
    unzip commandlinetools-linux-11076708_latest.zip && \
    mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    mv cmdline-tools ${ANDROID_SDK_ROOT}/cmdline-tools/latest && \
    rm commandlinetools-linux-11076708_latest.zip

# Accept Android SDK licenses
RUN yes | sdkmanager --licenses || true

# Install Android SDK components
# - platform-tools: includes adb
# - emulator: Android emulator
# - system-images: Android system image (using API 33 / Android 13 as default)
# - platforms: Android platform for building
RUN sdkmanager \
    "platform-tools" \
    "emulator" \
    "system-images;android-33;google_apis;x86_64" \
    "platforms;android-33" \
    "build-tools;33.0.0"

# Create a default AVD (can be overridden by user)
# This creates a baseline AVD that can be used if custom one is not provided
RUN echo "no" | avdmanager create avd \
    --force \
    --name "default_pixel_6" \
    --package "system-images;android-33;google_apis;x86_64" \
    --device "pixel_6" || true

# Install Python dependencies
COPY src/envs/android_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy OpenEnv core and android_env code
WORKDIR /app
COPY src/core/ /app/src/core/
COPY src/envs/android_env/ /app/src/envs/android_env/

# Create workspace directory for tasks and data
RUN mkdir -p /workspace/tasks /workspace/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose HTTP port
EXPOSE 8000

# Set up entrypoint script to handle emulator startup if needed
# Note: The emulator is started by android_env loader, not here
# We just run the FastAPI server

# Run server
CMD ["uvicorn", "envs.android_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
