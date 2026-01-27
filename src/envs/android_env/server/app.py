# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Android Environment.

This module creates an HTTP server that exposes the AndroidEnvironment
over HTTP endpoints, making it accessible via HTTPEnvClient.

The server is configured via environment variables:
    - ANDROID_AVD_NAME: Name of the Android Virtual Device (required)
    - ANDROID_TASK_PATH: Path to task textproto file (required)
    - ANDROID_ADB_PATH: Path to ADB (default: ~/Android/Sdk/platform-tools/adb)
    - ANDROID_EMULATOR_PATH: Path to emulator (default: ~/Android/Sdk/emulator/emulator)
    - ANDROID_AVD_HOME: AVD home directory (default: ~/.android/avd)
    - ANDROID_SDK_ROOT: SDK root directory (default: ~/Android/Sdk)
    - ANDROID_RUN_HEADLESS: Run headless (default: true)
    - ANDROID_IMAGE_FORMAT: Image encoding format (default: JPEG)
    - ANDROID_IMAGE_QUALITY: JPEG quality 1-100 (default: 85)

Usage:
    # Development (with environment variables):
    export ANDROID_AVD_NAME=Pixel_6_API_33
    export ANDROID_TASK_PATH=/workspace/tasks/my_task.textproto
    uvicorn envs.android_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.android_env.server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    python -m envs.android_env.server.app
"""

import os
from pathlib import Path

from core.env_server.http_server import create_app

from ..models import AndroidAction, AndroidObservation
from .android_environment import AndroidEnvironment

# Get configuration from environment variables
AVD_NAME = os.getenv("ANDROID_AVD_NAME")
TASK_PATH = os.getenv("ANDROID_TASK_PATH")
ADB_PATH = os.getenv("ANDROID_ADB_PATH", "~/Android/Sdk/platform-tools/adb")
EMULATOR_PATH = os.getenv(
    "ANDROID_EMULATOR_PATH", "~/Android/Sdk/emulator/emulator"
)
AVD_HOME = os.getenv("ANDROID_AVD_HOME", "~/.android/avd")
SDK_ROOT = os.getenv("ANDROID_SDK_ROOT", "~/Android/Sdk")
RUN_HEADLESS = os.getenv("ANDROID_RUN_HEADLESS", "true").lower() == "true"
IMAGE_FORMAT = os.getenv("ANDROID_IMAGE_FORMAT", "JPEG")
IMAGE_QUALITY = int(os.getenv("ANDROID_IMAGE_QUALITY", "85"))

# Validate required configuration
if not AVD_NAME:
    raise ValueError(
        "ANDROID_AVD_NAME environment variable is required. "
        "Set it to the name of your Android Virtual Device."
    )

if not TASK_PATH:
    raise ValueError(
        "ANDROID_TASK_PATH environment variable is required. "
        "Set it to the path of your task textproto file."
    )

# Expand paths
ADB_PATH = str(Path(ADB_PATH).expanduser())
EMULATOR_PATH = str(Path(EMULATOR_PATH).expanduser())
AVD_HOME = str(Path(AVD_HOME).expanduser())
SDK_ROOT = str(Path(SDK_ROOT).expanduser())
TASK_PATH = str(Path(TASK_PATH).expanduser())

print(f"Initializing Android Environment with:")
print(f"  AVD Name: {AVD_NAME}")
print(f"  Task Path: {TASK_PATH}")
print(f"  ADB Path: {ADB_PATH}")
print(f"  Emulator Path: {EMULATOR_PATH}")
print(f"  AVD Home: {AVD_HOME}")
print(f"  SDK Root: {SDK_ROOT}")
print(f"  Headless: {RUN_HEADLESS}")
print(f"  Image Format: {IMAGE_FORMAT} (Quality: {IMAGE_QUALITY})")

# Create the environment instance
env = AndroidEnvironment(
    task_path=TASK_PATH,
    avd_name=AVD_NAME,
    adb_path=ADB_PATH,
    emulator_path=EMULATOR_PATH,
    android_avd_home=AVD_HOME,
    android_sdk_root=SDK_ROOT,
    run_headless=RUN_HEADLESS,
    image_format=IMAGE_FORMAT,
    image_quality=IMAGE_QUALITY,
)

# Create the FastAPI app with web interface
app = create_app(env, AndroidAction, AndroidObservation, env_name="android_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
