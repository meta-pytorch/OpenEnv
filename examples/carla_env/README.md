# CARLA LLM Inference Examples

Run LLMs on CARLA autonomous driving scenarios from [sinatras/carla-env](https://blog.sinatras.dev/Carla-Env).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export HF_TOKEN="your-hf-token"
```

## Usage

### Trolley Problems

```bash
# Run single scenario
python trolley_problems.py --model claude-sonnet-4.5 --scenario footbridge

# Save camera images
python trolley_problems.py --model claude-sonnet-4.5 --scenario equal-1v1 --save-images

# Use HuggingFace Space
python trolley_problems.py --model gpt-5.2 --scenario saves-3v0 \
  --base-url https://sergiopaniego-carla-env.hf.space
```

### Maze Navigation

```bash
# Run navigation
python maze_navigation.py --model gpt-5.2

# Save images every 5 steps
python maze_navigation.py --model gpt-5.2 --save-images --image-interval 5
```

## Available Models

**Proprietary**: `claude-sonnet-4.5`, `claude-sonnet-4`, `gpt-4.1-mini`, `gpt-5.2`, `qwen3-max`

**Open (HF)**: `qwen2.5-72b`, `llama-3.3-70b`, `llama-3.1-70b`, `mixtral-8x7b`

## Available Scenarios

**Trolley Problems**: `equal-1v1`, `saves-3v0`, `deadzone-3v1`, `footbridge`, `self-sacrifice`, etc.

**Maze**: `maze-1` (153m navigation)

See `config.py` for full list.
