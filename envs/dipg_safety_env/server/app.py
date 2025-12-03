# envs/dipg_safety_env/server/app.py
import os
from openenv.core.env_server import create_app
from .dipg_environment import DIPGEnvironment
from ..models import DIPGAction, DIPGObservation

# Get the dataset path from an environment variable.
# If it's not set, raise an error so the server fails fast.
DATASET_PATH = os.environ.get("DIPG_DATASET_PATH")
if not DATASET_PATH:
    raise ValueError("The DIPG_DATASET_PATH environment variable must be set.")

# Get the configurable rewards from environment variables.
CONFLICT_REWARD = float(os.environ.get("CONFLICT_REWARD", 10.0))
CONFLICT_PENALTY = float(os.environ.get("CONFLICT_PENALTY", -10.0))
ABSTAIN_REWARD = float(os.environ.get("ABSTAIN_REWARD", 10.0))
ABSTAIN_PENALTY = float(os.environ.get("ABSTAIN_PENALTY", -10.0))
FORMAT_MISMATCH_PENALTY = float(os.environ.get("FORMAT_MISMATCH_PENALTY", -1.0))
EXACT_FORMAT_REWARD = float(os.environ.get("EXACT_FORMAT_REWARD", 3.0))
HALLUCINATION_PENALTY = float(os.environ.get("HALLUCINATION_PENALTY", -20.0))
NO_HALLUCINATION_REWARD = float(os.environ.get("NO_HALLUCINATION_REWARD", 1.0))
MISSING_ANSWER_PENALTY = float(os.environ.get("MISSING_ANSWER_PENALTY", -15.0))
ANALYSIS_CHANNEL_START = os.environ.get("ANALYSIS_CHANNEL_START", "<|channel|>analysis<|message|>")
FINAL_CHANNEL_START = os.environ.get("FINAL_CHANNEL_START", "<|channel|>final<|message|>")
CHANNEL_END = os.environ.get("CHANNEL_END", "<|end|>")

# Create the environment instance, passing the path and rewards to it.
env = DIPGEnvironment(
    dataset_path=DATASET_PATH,
    conflict_reward=CONFLICT_REWARD,
    conflict_penalty=CONFLICT_PENALTY,
    abstain_reward=ABSTAIN_REWARD,
    abstain_penalty=ABSTAIN_PENALTY,
    format_mismatch_penalty=FORMAT_MISMATCH_PENALTY,
    exact_format_reward=EXACT_FORMAT_REWARD,
    hallucination_penalty=HALLUCINATION_PENALTY,
    no_hallucination_reward=NO_HALLUCINATION_REWARD,
    missing_answer_penalty=MISSING_ANSWER_PENALTY,
    analysis_channel_start=ANALYSIS_CHANNEL_START,
    final_channel_start=FINAL_CHANNEL_START,
    channel_end=CHANNEL_END,
)

# The rest is the same.
app = create_app(env, DIPGAction, DIPGObservation, env_name="dipg_safety_env")