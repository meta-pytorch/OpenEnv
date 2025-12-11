#!/bin/bash
# Additional setup for chat_env
set -e

# Install Python dependencies
pip install --no-cache-dir -r /tmp/requirements.txt

# Set up cache directory for Hugging Face models
mkdir -p /.cache && chmod 777 /.cache

# Pre-download the GPT-2 model to avoid permission issues during runtime
python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')"
