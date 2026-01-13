# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Android Environment Server Implementation with complete features.

This module wraps DeepMind's android_env with:
- Full gesture support (tap, swipe, scroll, etc.)
- ADB integration for text input and button presses
- Shared memory optimization for parallel training
- Gesture sequencing
"""

import base64
import io
import logging
import subprocess
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from android_env import loader
from android_env.components import config_classes
from android_env.proto import adb_pb2
from dm_env import specs
from PIL import Image

from core.env_server.interfaces import Environment
from core.env_server.types import State

from ..models import AndroidAction, AndroidObservation
from .gestures import ADBCommands, GestureBuilder

logger = logging.getLogger(__name__)


class AndroidEnvironment(Environment):
    """
    Enhanced Android environment wrapper for OpenEnv.

    Features:
    - Complete gesture support (swipe, scroll, long press, etc.)
    - ADB text input and button press
    - Gesture sequencing (multi-step gestures)
    - Optional shared memory for high-performance deployments
    - Action buffering for gesture composition
    """

    def __init__(
        self,
        task_path: str,
        avd_name: str,
        adb_path: str = "~/Android/Sdk/platform-tools/adb",
        emulator_path: str = "~/Android/Sdk/emulator/emulator",
        android_avd_home: str = "~/.android/avd",
        android_sdk_root: str = "~/Android/Sdk",
        run_headless: bool = True,
        image_format: str = "JPEG",
        image_quality: int = 85,
        use_shared_memory: bool = False,
        shared_memory_name: Optional[str] = None,
    ):
        """Initialize the Android environment.

        Args:
            task_path: Path to the android_env task textproto file.
            avd_name: Name of the Android Virtual Device to use.
            adb_path: Path to the ADB executable.
            emulator_path: Path to the Android emulator executable.
            android_avd_home: Path to the AVD home directory.
            android_sdk_root: Path to the Android SDK root.
            run_headless: Whether to run the emulator in headless mode.
            image_format: Format for encoding screen images ("JPEG" or "PNG").
            image_quality: Quality for JPEG encoding (1-100).
            use_shared_memory: Use shared memory for zero-copy observations.
            shared_memory_name: Name for shared memory segment.
        """
        super().__init__()

        self._task_path = task_path
        self._avd_name = avd_name
        self._adb_path = adb_path
        self._image_format = image_format
        self._image_quality = image_quality
        self._use_shared_memory = use_shared_memory

        # Gesture sequencing state
        self._gesture_queue: List[dict] = []
        self._executing_gesture = False

        # Create android_env configuration
        config = config_classes.AndroidEnvConfig(
            task=config_classes.FilesystemTaskConfig(path=task_path),
            simulator=config_classes.EmulatorConfig(
                emulator_launcher=config_classes.EmulatorLauncherConfig(
                    emulator_path=emulator_path,
                    android_sdk_root=android_sdk_root,
                    android_avd_home=android_avd_home,
                    avd_name=avd_name,
                    run_headless=run_headless,
                ),
                adb_controller=config_classes.AdbControllerConfig(adb_path=adb_path),
            ),
        )

        # Load the android_env environment
        logger.info(f"Loading Android environment with AVD: {avd_name}")
        self._android_env = loader.load(config)

        # Get action and observation specs
        self._action_spec = self._android_env.action_spec()
        self._observation_spec = self._android_env.observation_spec()

        # Get screen dimensions from first observation
        initial_obs = self._android_env.reset().observation
        pixels = initial_obs.get("pixels")
        if pixels is not None:
            self._screen_height, self._screen_width, _ = pixels.shape
        else:
            self._screen_height, self._screen_width = 1920, 1080  # Default

        # Set up shared memory if requested
        self._shared_mem = None
        if use_shared_memory:
            mem_size = self._screen_height * self._screen_width * 3  # RGB
            self._shared_mem_name = shared_memory_name or f"android_env_{uuid4().hex[:8]}"
            try:
                self._shared_mem = shared_memory.SharedMemory(
                    name=self._shared_mem_name,
                    create=True,
                    size=mem_size
                )
                logger.info(f"Created shared memory: {self._shared_mem_name}")
            except Exception as e:
                logger.warning(f"Could not create shared memory: {e}. Falling back to encoding.")
                self._use_shared_memory = False

        # Initialize state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._latest_timestep = None

        logger.info(f"Android environment initialized successfully")
        logger.info(f"Screen size: {self._screen_width}x{self._screen_height}")
        logger.info(f"Action spec: {list(self._action_spec.keys())}")

    def reset(self) -> AndroidObservation:
        """Reset the Android environment for a new episode."""
        logger.info("Resetting Android environment...")

        # Clear gesture queue
        self._gesture_queue = []
        self._executing_gesture = False

        # Reset android_env
        self._latest_timestep = self._android_env.reset()

        # Update state
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Convert timestep to observation
        observation = self._convert_timestep_to_observation(self._latest_timestep)

        logger.info(f"Reset complete. Episode ID: {self._state.episode_id}")
        return observation

    def step(self, action: AndroidAction) -> AndroidObservation:  # type: ignore[override]
        """Execute an action in the Android environment."""
        # Convert OpenEnv action to gesture sequence or direct action
        gesture_actions = self._convert_action_to_gestures(action)

        # Execute all actions in the gesture sequence
        for i, gesture_action in enumerate(gesture_actions):
            android_action = self._create_android_action(gesture_action)
            self._latest_timestep = self._android_env.step(android_action)

            # Update state on last action of sequence
            if i == len(gesture_actions) - 1:
                self._state.step_count += 1

        # Convert final timestep to observation
        observation = self._convert_timestep_to_observation(self._latest_timestep)

        # Check if episode is done
        if self._latest_timestep.last():
            observation.done = True
            logger.info(f"Episode ended after {self._state.step_count} steps")

        return observation

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up the Android environment."""
        logger.info("Closing Android environment...")
        if hasattr(self, "_android_env"):
            self._android_env.close()
        if self._shared_mem:
            try:
                self._shared_mem.close()
                self._shared_mem.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up shared memory: {e}")
        logger.info("Android environment closed")

    def _convert_action_to_gestures(self, action: AndroidAction) -> List[dict]:
        """Convert high-level action to sequence of primitive gestures."""
        tool_name = action.tool_name
        params = action.parameters

        # Use GestureBuilder for complex gestures
        if tool_name == "tap":
            return GestureBuilder.tap(params["x"], params["y"])

        elif tool_name == "swipe":
            return GestureBuilder.swipe(
                params["x1"], params["y1"],
                params["x2"], params["y2"],
                params.get("duration_ms", 300)
            )

        elif tool_name == "long_press":
            return GestureBuilder.long_press(
                params["x"], params["y"],
                params.get("duration_ms", 1000)
            )

        elif tool_name == "double_tap":
            return GestureBuilder.double_tap(params["x"], params["y"])

        elif tool_name == "scroll_down":
            return GestureBuilder.scroll_down(
                params.get("x", 0.5),
                params.get("distance", 0.5)
            )

        elif tool_name == "scroll_up":
            return GestureBuilder.scroll_up(
                params.get("x", 0.5),
                params.get("distance", 0.5)
            )

        elif tool_name == "swipe_left":
            return GestureBuilder.swipe_left(
                params.get("y", 0.5),
                params.get("distance", 0.5)
            )

        elif tool_name == "swipe_right":
            return GestureBuilder.swipe_right(
                params.get("y", 0.5),
                params.get("distance", 0.5)
            )

        elif tool_name == "type_text":
            # Execute ADB text input command
            self._execute_adb_text(params["text"])
            # Return a no-op touch action
            return [{"action_type": 2, "x": 0.5, "y": 0.5, "duration_ms": 100}]

        elif tool_name == "press_button":
            # Execute ADB keyevent command
            self._execute_adb_button(params["button"])
            # Return a no-op touch action
            return [{"action_type": 2, "x": 0.5, "y": 0.5, "duration_ms": 100}]

        else:
            raise ValueError(f"Unknown action tool_name: {tool_name}")

    def _create_android_action(self, gesture_action: dict) -> Dict[str, np.ndarray]:
        """Create android_env action from gesture primitive."""
        action = {}
        action_type = gesture_action["action_type"]
        x = gesture_action["x"]
        y = gesture_action["y"]

        for key, spec in self._action_spec.items():
            if key == "action_type":
                action[key] = np.array(action_type, dtype=spec.dtype)
            elif key == "touch_position":
                action[key] = np.array([np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)], dtype=spec.dtype)
            else:
                # Fill other fields with defaults
                if isinstance(spec, specs.DiscreteArray):
                    action[key] = np.array(0, dtype=spec.dtype)
                else:
                    action[key] = np.zeros(spec.shape, dtype=spec.dtype)

        return action

    def _execute_adb_text(self, text: str) -> None:
        """Execute ADB text input command."""
        try:
            cmd = ADBCommands.text_input(text)
            adb_request = adb_pb2.AdbRequest()
            adb_request.generic.command = cmd
            self._android_env.execute_adb_call(adb_request)
            logger.info(f"Executed ADB text input: {text[:20]}...")
        except Exception as e:
            logger.error(f"ADB text input failed: {e}")

    def _execute_adb_button(self, button: str) -> None:
        """Execute ADB button press command."""
        try:
            # Map common button names to keycodes
            button_map = {
                "HOME": ADBCommands.KEYCODE_HOME,
                "BACK": ADBCommands.KEYCODE_BACK,
                "MENU": ADBCommands.KEYCODE_MENU,
                "ENTER": ADBCommands.KEYCODE_ENTER,
                "SEARCH": ADBCommands.KEYCODE_SEARCH,
                "DELETE": ADBCommands.KEYCODE_DEL,
                "TAB": ADBCommands.KEYCODE_TAB,
                "SPACE": ADBCommands.KEYCODE_SPACE,
            }
            keycode = button_map.get(button.upper(), button)

            cmd = ADBCommands.keyevent(keycode)
            adb_request = adb_pb2.AdbRequest()
            adb_request.generic.command = cmd
            self._android_env.execute_adb_call(adb_request)
            logger.info(f"Executed ADB button press: {button}")
        except Exception as e:
            logger.error(f"ADB button press failed: {e}")

    def _convert_timestep_to_observation(self, timestep: Any) -> AndroidObservation:
        """Convert android_env TimeStep to AndroidObservation."""
        obs_dict = timestep.observation
        pixels = obs_dict.get("pixels")

        if pixels is None:
            raise ValueError("No pixels found in android_env observation")

        height, width, channels = pixels.shape

        # Handle observation encoding
        if self._use_shared_memory and self._shared_mem:
            # Write pixels to shared memory
            screen_image_b64 = self._write_to_shared_memory(pixels)
        else:
            # Encode to base64
            screen_image_b64 = self._encode_image(pixels)

        # Extract extras
        extras = {k: v for k, v in obs_dict.items() if k != "pixels"}
        if hasattr(self._android_env, "task_extras"):
            task_extras = self._android_env.task_extras(latest_only=True)
            extras.update({"task_extras": task_extras})

        observation = AndroidObservation(
            screen_image=screen_image_b64,
            screen_width=width,
            screen_height=height,
            timestamp_ms=int(time.time() * 1000),
            orientation=0,
            pixels_shape=(height, width, channels),
            extras=extras,
            done=timestep.last(),
            reward=float(timestep.reward) if timestep.reward is not None else 0.0,
        )

        return observation

    def _encode_image(self, pixels: np.ndarray) -> str:
        """Encode numpy pixel array to base64 string."""
        image = Image.fromarray(pixels.astype(np.uint8))
        buffer = io.BytesIO()

        if self._image_format == "JPEG":
            image.save(buffer, format="JPEG", quality=self._image_quality)
        elif self._image_format == "PNG":
            image.save(buffer, format="PNG")
        else:
            raise ValueError(f"Unsupported image format: {self._image_format}")

        buffer.seek(0)
        image_bytes = buffer.read()
        return base64.b64encode(image_bytes).decode("utf-8")

    def _write_to_shared_memory(self, pixels: np.ndarray) -> str:
        """Write pixels to shared memory and return memory name."""
        if not self._shared_mem:
            return self._encode_image(pixels)  # Fallback

        try:
            # Write pixels directly to shared memory
            np_array = np.ndarray(
                pixels.shape,
                dtype=pixels.dtype,
                buffer=self._shared_mem.buf
            )
            np_array[:] = pixels[:]
            # Return shared memory name instead of image data
            return f"shm://{self._shared_mem_name}"
        except Exception as e:
            logger.error(f"Shared memory write failed: {e}, falling back to encoding")
            return self._encode_image(pixels)

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
