#!/usr/bin/env python3
"""
Play any Atari game using a Vision-Language Model via the Hugging Face Router API.

The script:
1. Starts an Atari environment (Docker) for the selected game
2. Sends recent screen frames to a vision-language model
3. Parses the model's integer response into an Atari action id
4. Reports a minimal summary

Notes:
- Frames are sent raw (no overlays, cropping, or resizing)
- The model receives the legal action ids each step and must return one integer

Usage:
    export API_KEY=your_hf_token_here
    python examples/atari_pong_inference.py --game breakout --model Qwen/Qwen3-VL-8B-Instruct:novita
"""

import os
import re
import base64
import gradio as gr
from collections import deque
from io import BytesIO
from typing import Deque, List, Optional

import numpy as np
from PIL import Image
from openai import OpenAI

from envs.atari_env import AtariEnv, AtariAction


# API Configuration
# For HuggingFace: Use HF_TOKEN and set API_BASE_URL
API_BASE_URL = "https://router.huggingface.co/v1"  # Hugging Face Router endpoint
API_KEY = os.getenv("API_KEY")  # Required for Hugging Face
ATARI_ENV_BASE_URL = os.getenv("ATARI_ENV_BASE_URL")  # Optional: connect to a remote Atari env

# Vision-Language Model (Hugging Face Router compatible)
MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"

# Configuration
TEMPERATURE = 0.7
MAX_STEPS_PER_GAME = 10000
MAX_TOKENS = 16
VERBOSE = True
FRAME_HISTORY_LENGTH = 4
DISPLAY_SCALE = 3  # Scale factor for enlarging frames sent to UI
MODEL_SCALE = 3    # Scale factor for enlarging frames sent to the model

# Generic game prompt for the vision model
VISION_PROMPT = (
    "You are playing an Atari-style game. You will be given recent frames "
    "and the list of legal action ids for the current step. "
    "Respond with a single integer that is exactly one of the legal action ids. "
    "Do not include any words or punctuation — only the integer."
)

ACTIONS_LOOKUP = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

def screen_to_base64(screen: List[int], screen_shape: List[int]) -> str:
    """Convert flattened screen array to base64 encoded PNG image (no processing)."""
    screen_array = np.array(screen, dtype=np.uint8).reshape(screen_shape)
    image = Image.fromarray(screen_array)
    # Enlarge image for model input if configured
    try:
        if MODEL_SCALE and MODEL_SCALE > 1:
            image = image.resize((image.width * MODEL_SCALE, image.height * MODEL_SCALE), Image.NEAREST)
    except Exception:
        pass
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def screen_to_numpy(screen: List[int], screen_shape: List[int]) -> np.ndarray:
    """Convert flattened screen to a larger RGB numpy array for gr.Image display."""
    arr = np.array(screen, dtype=np.uint8).reshape(screen_shape)
    # Let Pillow infer mode to avoid deprecation warnings about the 'mode' parameter
    img = Image.fromarray(arr)
    # Enlarge with nearest-neighbor to preserve pixel edges
    try:
        img = img.resize((img.width * DISPLAY_SCALE, img.height * DISPLAY_SCALE), Image.NEAREST)
    except Exception:
        pass
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def content_text(text: str) -> dict:
    return {"type": "text", "text": text}


def content_image_b64(b64_png: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}


def build_messages(prompt: str, frame_history_b64: Deque[str], current_b64: str, legal_actions: List[int]) -> List[dict]:
    messages: List[dict] = [
        {"role": "system", "content": [content_text(prompt)]}
    ]
    if len(frame_history_b64) > 1:
        total = len(frame_history_b64)
        messages.extend([
            {
                "role": "user",
                "content": [
                    content_text(f"Frame -{total - idx}"),
                    content_image_b64(_img),
                ],
            }
            for idx, _img in enumerate(list(frame_history_b64)[:-1])
        ])
    messages.append({
        "role": "user",
        "content": [content_text("Current frame:"), content_image_b64(current_b64)],
    })
    # Include mapping of action ids to human-readable names for the model
    action_pairs = ", ".join([f"{aid}:{ACTIONS_LOOKUP.get(aid, 'UNK')}" for aid in legal_actions])
    messages.append({
        "role": "user",
        "content": [content_text(f"Legal actions (id:name): {action_pairs}. Respond with exactly one INTEGER id.")],
    })
    return messages


class GameSession:
    """Holds environment/model state and advances one step per tick."""
    def __init__(self, game: str, model_name: str, prompt_text: str):
        if not API_KEY:
            raise RuntimeError("Missing API_KEY for HF Router")
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.env: Optional[AtariEnv] = None
        self.model_name = model_name
        self.game = game
        self.prompt = (prompt_text or "").strip() or VISION_PROMPT
        self.frame_history_base64: Deque[str] = deque(maxlen=FRAME_HISTORY_LENGTH)
        self.total_reward = 0.0
        self.steps = 0
        self.done = False

        # Start environment
        self.env = AtariEnv(base_url=f"https://burtenshaw-{game}.hf.space")
        result = self.env.reset()
        self.obs = result.observation
        self.log_message = f"Game: {self.game} started"

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None
        self.done = True

    def next_frame(self) -> Optional[np.ndarray]:
        # Snapshot env reference to avoid race if another thread closes it mid-tick
        env = self.env
        if self.done or env is None:
            return None
        if self.steps >= MAX_STEPS_PER_GAME:
            self.close()
            return None

        # Prepare images
        image_data = screen_to_base64(self.obs.screen, self.obs.screen_shape)
        if FRAME_HISTORY_LENGTH > 0:
            self.frame_history_base64.append(image_data)

        # Build messages (deduplicated helpers)
        messages = build_messages(self.prompt, self.frame_history_base64, image_data, self.obs.legal_actions)

        # Query model
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
            action_id = parse_action(response_text, self.obs.legal_actions)
        except Exception:
            action_id = 0 if 0 in self.obs.legal_actions else self.obs.legal_actions[0]

        # Step env (guard against races with stop/close)
        try:
            result = env.step(AtariAction(action_id=action_id))
        except AttributeError:
            # env likely closed concurrently
            self.close()
            return None
        except Exception:
            # Network/server error - stop session gracefully
            self.close()
            return None
        self.obs = result.observation
        self.total_reward += result.reward or 0.0
        self.steps += 1
        if result.done:
            self.done = True
            self.close()
        
        action_name = ACTIONS_LOOKUP.get(action_id, str(action_id))
        self.log_message += f"\nAction: {action_name} ({action_id}) Reward: {result.reward}"
        return screen_to_numpy(self.obs.screen, self.obs.screen_shape)


def parse_action(text: str, legal_actions: List[int]) -> int:
    """
    Parse action from model output.
    Handles chain-of-thought format by taking the LAST valid number found.
    
    Args:
        text: Model's text response (may include reasoning)
        legal_actions: List of valid action IDs
    
    Returns:
        Selected action ID (defaults to NOOP if parsing fails)
    """
    # Look for single digit numbers in the response
    numbers = re.findall(r'\b\d+\b', text)
    
    # Check from the end (last number is likely the final action after reasoning)
    for num_str in reversed(numbers):
        action_id = int(num_str)
        if action_id in legal_actions:
            return action_id
    
    # Default to NOOP if available, otherwise first legal action
    return 0 if 0 in legal_actions else legal_actions[0]


# Legacy CLI loop removed; Gradio's Image.every drives stepping via GameSession.next_frame


def start_session(game: str, model_name: str, prompt_text: str) -> Optional[GameSession]:
    try:
        return GameSession(game=game, model_name=model_name, prompt_text=prompt_text)
    except Exception as e:
        raise gr.Error(str(e))


def stop_session(session: Optional[GameSession]) -> Optional[GameSession]:
    if isinstance(session, GameSession):
        session.close()
    return None


def frame_tick(session: Optional[GameSession]) -> Optional[np.ndarray]:
    if not isinstance(session, GameSession):
        return None
    frame = session.next_frame()
    if frame is None:
        # Auto-stop when done
        session.close()
        return None
    return frame


def log_tick(session: Optional[GameSession]) -> str:
    if not isinstance(session, GameSession):
        return ""
    return session.log_message


def launch_gradio_app():
    games = [
        "pong",
        "breakout",
        "pacman",
    ]
    models = [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-72B-A14B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
    ]

    with gr.Blocks() as demo:
        gr.Markdown("""
        ### Atari Vision-Language Control
        - Select a game and model, edit the prompt, then click Start.
        - Frames are streamed directly from the environment without modification.
        - There are a limited number of environment spaces via `"https://burtenshaw-{game}.hf.space"`
        - Duplicate the space and change environment variables if you want to use a different game.
        """)


        session_state = gr.State()
        
        with gr.Row():
        
            with gr.Column():
                game_dd = gr.Dropdown(choices=games, value="pong", label="Game")
                model_dd = gr.Dropdown(choices=models, value=models[0], label="Model")
                prompt_tb = gr.Textbox(label="Prompt", value=VISION_PROMPT, lines=6)
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop")

            with gr.Column():
                out_image = gr.Image(label="Game Stream", type="numpy", value=frame_tick, inputs=[session_state], every=0.1, height=480, width=640)
        
        out_text = gr.Textbox(label="Game Logs", value=log_tick, inputs=[session_state], lines=10, every=0.5)
        
        # Controls
        start_btn.click(start_session, inputs=[game_dd, model_dd, prompt_tb], outputs=[session_state])
        stop_btn.click(stop_session, inputs=[session_state], outputs=[session_state])

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    launch_gradio_app()
