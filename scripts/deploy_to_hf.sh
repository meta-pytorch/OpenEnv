#!/bin/bash

# OpenEnv Hugging Face Deployment Preparation Script
# This script prepares files for deployment to Hugging Face Spaces

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/deploy_to_hf.sh --env <environment_name> [options]

Required arguments:
  --env <name>               Environment name under src/envs (e.g. textarena_env)

Optional arguments:
  --base-sha <sha|tag>       Override openenv-base image reference (defaults to :latest)
  --hf-namespace <user>      Hugging Face username/organization (defaults to HF_USERNAME or meta-openenv)
  --staging-dir <path>       Output directory for staging (defaults to hf-staging)
  --space-suffix <suffix>    Suffix to add to space name (e.g., "-test" for test spaces)
  --private                   Deploy the space as private (default: public)
  --dry-run                  Prepare files without pushing to Hugging Face Spaces
  -h, --help                 Show this help message

Positional compatibility:
  You can also call the script as:
    scripts/deploy_to_hf.sh <env_name> [base_image_sha]

Examples:
  scripts/deploy_to_hf.sh --env textarena_env --hf-namespace my-team
  scripts/deploy_to_hf.sh echo_env --private --hf-namespace my-org
EOF
}

sed_in_place() {
    local expression="$1"
    local target_file="$2"
    if sed --version >/dev/null 2>&1; then
        sed -i "$expression" "$target_file"
    else
        sed -i '' "$expression" "$target_file"
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Detect if we're running in GitHub Actions
IS_GITHUB_ACTIONS=false
if [ -n "${GITHUB_ACTIONS:-}" ]; then
    IS_GITHUB_ACTIONS=true
fi

# Check for hf CLI - but allow dry-run mode to work without it
if ! command -v hf >/dev/null 2>&1; then
    echo "Warning: huggingface-hub CLI 'hf' not found in PATH." >&2
    echo "Install the HF CLI: curl -LsSf https://hf.co/cli/install.sh | sh" >&2
    if [ "${1:-}" != "--dry-run" ] && [ "${2:-}" != "--dry-run" ]; then
        echo "Error: hf is required for deployment (use --dry-run to skip deployment)" >&2
        exit 1
    fi
fi

ENV_NAME=""
BASE_IMAGE_SHA=""
HF_NAMESPACE="${HF_NAMESPACE:-}"  # Initialize from env var if set, otherwise empty
STAGING_DIR="hf-staging"
SPACE_SUFFIX=""
PRIVATE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            ENV_NAME="$2"
            shift 2
            ;;
        --hub-tag)
            HUB_TAG="$2"
            shift 2
            ;;
        --base-sha|--base-image-sha)
            BASE_IMAGE_SHA="$2"
            shift 2
            ;;
        --namespace|--hf-namespace|--hf-user|--hf-username)
            HF_NAMESPACE="$2"
            shift 2
            ;;
        --staging-dir)
            STAGING_DIR="$2"
            shift 2
            ;;
        --suffix|--space-suffix)
            SPACE_SUFFIX="$2"
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -* )
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [ -z "$ENV_NAME" ]; then
                ENV_NAME="$1"
            elif [ -z "$BASE_IMAGE_SHA" ]; then
                BASE_IMAGE_SHA="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$HUB_TAG" ]; then
    HUB_TAG="openenv"
fi

if [ -z "$ENV_NAME" ]; then
    echo "Error: Environment name is required" >&2
    usage
    exit 1
fi

if [[ "$ENV_NAME" == *","* || "$ENV_NAME" == *" "* ]]; then
    echo "Error: only one environment can be deployed per invocation (received '$ENV_NAME')." >&2
    exit 1
fi

if [ ! -d "src/envs/$ENV_NAME" ]; then
    echo "Error: Environment '$ENV_NAME' not found under src/envs" >&2
    exit 1
fi

# Try to get HF_USERNAME, but handle failures gracefully (especially in CI before auth)
if command -v hf >/dev/null 2>&1; then
    HF_USERNAME=$(hf auth whoami 2>/dev/null | head -n1 | tr -d '\n' || echo "")
fi

if [ -z "$HF_NAMESPACE" ]; then
    # Check HF_USERNAME (env var or detected from CLI)
    if [ -n "${HF_USERNAME:-}" ]; then
        HF_NAMESPACE="${HF_USERNAME}"
    else
        HF_NAMESPACE="meta-openenv"
    fi
fi

echo "🙋 Using namespace: $HF_NAMESPACE. You can override with --hf-namespace"

# Set base image reference (using GHCR)
if [ -n "$BASE_IMAGE_SHA" ]; then
    BASE_IMAGE_REF="ghcr.io/meta-pytorch/openenv-base:$BASE_IMAGE_SHA"
    echo "Using specific SHA for openenv-base: $BASE_IMAGE_SHA"
else
    BASE_IMAGE_REF="ghcr.io/meta-pytorch/openenv-base:latest"
fi

# Create staging directory
CURRENT_STAGING_DIR="${STAGING_DIR}/${HF_NAMESPACE}/${ENV_NAME}"
# Ensure clean staging directory
rm -rf "$CURRENT_STAGING_DIR"
mkdir -p "$CURRENT_STAGING_DIR/src/core"
mkdir -p "$CURRENT_STAGING_DIR/src/envs/$ENV_NAME"

# Copy core files
cp -R src/core/* "$CURRENT_STAGING_DIR/src/core/"

# Copy environment files
cp -R src/envs/$ENV_NAME/* "$CURRENT_STAGING_DIR/src/envs/$ENV_NAME/"

echo "📁 Copied core and $ENV_NAME environment files to $CURRENT_STAGING_DIR"

# Create environment-specific multi-stage Dockerfile
create_environment_dockerfile() {
    local env_name=$1
    
    # Create base Dockerfile
    cat > "$CURRENT_STAGING_DIR/Dockerfile" << DOCKERFILE_EOF
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the specified openenv-base image
FROM $BASE_IMAGE_REF
DOCKERFILE_EOF

    # Add environment-specific dependencies
    case $env_name in
        "echo_env")
            # Echo environment needs no additional dependencies
            ;;
        "coding_env")
            cat >> "$CURRENT_STAGING_DIR/Dockerfile" << 'DOCKERFILE_EOF'
# Install smolagents for code execution
RUN pip install --no-cache-dir smolagents
DOCKERFILE_EOF
            ;;
        "chat_env")
            cat >> "$CURRENT_STAGING_DIR/Dockerfile" << 'DOCKERFILE_EOF'
# Install additional dependencies for ChatEnvironment
RUN pip install --no-cache-dir torch transformers

# Set up cache directory for Hugging Face models
RUN mkdir -p /.cache && chmod 777 /.cache
ENV HF_HOME=/.cache
ENV TRANSFORMERS_CACHE=/.cache

# Pre-download the GPT-2 model to avoid permission issues during runtime
RUN python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')"
DOCKERFILE_EOF
            ;;
        "atari_env")
            cat >> "$CURRENT_STAGING_DIR/Dockerfile" << 'DOCKERFILE_EOF'
# Install ALE-specific dependencies
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    ale-py>=0.8.0 \
    numpy>=1.24.0
DOCKERFILE_EOF
            ;;
        "textarena_env")
            cat >> "$CURRENT_STAGING_DIR/Dockerfile" << 'DOCKERFILE_EOF'
# Install system libraries required by TextArena
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install TextArena and supporting Python packages
RUN pip install --no-cache-dir \
    textarena==0.6.1 \
    nltk==3.9.2
DOCKERFILE_EOF
            ;;
        "openspiel_env")
            # OpenSpiel requires special C++ build process - replace entire Dockerfile
            cat > "$CURRENT_STAGING_DIR/Dockerfile" << DOCKERFILE_EOF
# OpenSpiel environment using pre-built OpenSpiel base image
ARG OPENSPIEL_BASE_IMAGE=ghcr.io/meta-pytorch/openenv-openspiel-base:sha-e622c7e
FROM \${OPENSPIEL_BASE_IMAGE}

# Copy OpenEnv core (base image already set WORKDIR=/app)
WORKDIR /app
COPY src/core/ /app/src/core/

# Copy OpenSpiel environment
COPY src/envs/openspiel_env/ /app/src/envs/openspiel_env/

# Extend Python path for OpenEnv (base image set PYTHONPATH=/app/src)
# We prepend OpenSpiel paths
ENV PYTHONPATH=/repo:/repo/build/python:/app/src

# OpenSpiel-specific environment variables (can be overridden at runtime)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Health check (curl is provided by openenv-base)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Note: EXPOSE 8000 already set by openenv-base

# Run the FastAPI server (uvicorn installed by openenv-base)
CMD ["uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE_EOF
            echo "Created special OpenSpiel Dockerfile with C++ build process"
            echo "OpenSpiel builds can take 10-15 minutes due to C++ compilation"
            return  # Skip the common parts since OpenSpiel has its own complete Dockerfile
            ;;
    esac

    # Add common parts
    cat >> "$CURRENT_STAGING_DIR/Dockerfile" << 'DOCKERFILE_EOF'

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/ENV_NAME_PLACEHOLDER/ /app/src/envs/ENV_NAME_PLACEHOLDER/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.ENV_NAME_PLACEHOLDER.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE_EOF

    # Replace placeholder with actual environment name
    sed_in_place "s/ENV_NAME_PLACEHOLDER/$env_name/g" "$CURRENT_STAGING_DIR/Dockerfile"
}

create_environment_dockerfile "$ENV_NAME"

# Add web interface support
echo "ENV ENABLE_WEB_INTERFACE=true" >> $CURRENT_STAGING_DIR/Dockerfile

# Create environment-specific README
create_readme() {
    local env_name=$1
    
    # Capitalize first letter of environment name
    env_title=$(echo "$env_name" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
    
    # Set environment-specific colors and emoji
    case $env_name in
        "atari_env")
            EMOJI="🕹️"
            COLOR_FROM="red"
            COLOR_TO="yellow"
            ;;
        "coding_env")
            EMOJI="💻"
            COLOR_FROM="blue"
            COLOR_TO="gray"
            ;;
        "openspiel_env")
            EMOJI="🎮"
            COLOR_FROM="purple"
            COLOR_TO="indigo"
            ;;
        "echo_env")
            EMOJI="🔊"
            COLOR_FROM="blue"
            COLOR_TO="gray"
            ;;
        "chat_env")
            EMOJI="💬"
            COLOR_FROM="blue"
            COLOR_TO="green"
            ;;
        "textarena_env")
            EMOJI="📜"
            COLOR_FROM="green"
            COLOR_TO="blue"
            ;;
        *)
            EMOJI="🐳"
            COLOR_FROM="blue"
            COLOR_TO="green"
            ;;
    esac

    cat > "$CURRENT_STAGING_DIR/README.md" << README_EOF
---
title: ${env_title} Environment Server
emoji: ${EMOJI}
colorFrom: ${COLOR_FROM}
colorTo: ${COLOR_TO}
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - ${HUB_TAG}
---

# ${env_title} Environment Server

FastAPI server for ${env_name} environment powered by Meta's OpenEnv.

## About

This Space provides a containerized environment for ${env_name} interactions.
Built with FastAPI and OpenEnv framework.

## Web Interface

This deployment includes an interactive web interface for exploring the environment:
- **HumanAgent Interface**: Interact with the environment using a web form
- **State Observer**: Real-time view of environment state and action history
- **Live Updates**: WebSocket-based real-time updates

Access the web interface at: \`/web\`

README_EOF

    # Add environment-specific information
    case $env_name in
        "echo_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## Echo Environment

Simple test environment that echoes back messages. Perfect for testing the OpenEnv APIs.

### Usage
Send a POST request to `/step` with:
```json
{
  "message": "Hello World"
}
```
README_EOF
            ;;
        "coding_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## Coding Environment

Executes Python code in a sandboxed environment with safety checks.

### Usage
Send a POST request to `/step` with:
```json
{
  "code": "print('Hello World')"
}
```
README_EOF
            ;;
        "chat_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## Chat Environment

Provides a chat-based interface for LLMs with tokenization support.

### Usage
Send a POST request to `/step` with tokenized input:
```json
{
  "tokens": [1, 2, 3, 4, 5]
}
```
README_EOF
            ;;
        "atari_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## Atari Environment

Provides Atari 2600 games via the Arcade Learning Environment (ALE).

### Usage
Send a POST request to `/step` with:
```json
{
  "action_id": 0,
  "game_name": "pong"
}
```
README_EOF
            ;;
        "openspiel_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## OpenSpiel Environment

Provides access to OpenSpiel games for multi-agent reinforcement learning.

### Usage
Send a POST request to `/step` with:
```json
{
  "action": {
    "action_id": 1
  }
}
```
README_EOF
            ;;
        "textarena_env")
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'
## TextArena Environment

Runs TextArena games such as Wordle, GuessTheNumber, or Chess through a unified HTTP API.

### Usage
Send a POST request to `/step` with:
```json
{
  "message": "raise shield"
}
```
README_EOF
            ;;
        *)
            cat >> "$CURRENT_STAGING_DIR/README.md" << 'README_EOF'

## API Documentation

Visit `/docs` for interactive API documentation.

## Health Check

The environment provides a health check endpoint at `/health`.
README_EOF
            ;;
    esac
}

create_readme "$ENV_NAME"
echo "📝 Created README and web interface support for HF Space"

if $DRY_RUN; then
    echo "👀 Dry run enabled; skipping Hugging Face upload."
    exit 0
fi

echo "🔑 Ensuring Hugging Face authentication..."

# Set up authentication based on environment
TOKEN_ARGS=()
if [ "$IS_GITHUB_ACTIONS" = true ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN secret is required in GitHub Actions" >&2
        echo "Please set the HF_TOKEN secret in repository settings" >&2
        exit 1
    fi
    echo "Using HF_TOKEN from GitHub Actions environment"
    # In CI, pass token directly to commands via --token flag
    TOKEN_ARGS=(--token "$HF_TOKEN")
elif [ -n "${HF_TOKEN:-}" ]; then
    # If HF_TOKEN is set locally, use it
    echo "Using HF_TOKEN environment variable"
    TOKEN_ARGS=(--token "$HF_TOKEN")
else
    # Interactive mode: check if user is authenticated
    if ! hf auth whoami >/dev/null 2>&1; then
        echo "Not authenticated. Please login to Hugging Face..."
        hf auth login
        if ! hf auth whoami >/dev/null 2>&1; then
            echo "Error: Hugging Face authentication failed" >&2
            exit 1
        fi
    fi
fi

# Verify authentication works (skip in CI if using token directly)
if [ ${#TOKEN_ARGS[@]} -eq 0 ]; then
    if ! hf auth whoami >/dev/null 2>&1; then
        echo "Error: Not authenticated with Hugging Face" >&2
        echo "Run 'hf auth login' or set HF_TOKEN environment variable" >&2
        exit 1
    fi
    CURRENT_USER=$(hf auth whoami | head -n1 | tr -d '\n')
    echo "✅ Authenticated as: $CURRENT_USER"
    if [ "$CURRENT_USER" != "$HF_NAMESPACE" ]; then
        echo "⚠️  Deploying to namespace '$HF_NAMESPACE' (different from your user '$CURRENT_USER')"
    fi
else
    echo "✅ Token configured for deployment"
fi

SPACE_REPO="${HF_NAMESPACE}/${ENV_NAME}${SPACE_SUFFIX}"

# Get absolute path to staging directory
if [ ! -d "$CURRENT_STAGING_DIR" ]; then
    echo "Error: Staging directory not found: $CURRENT_STAGING_DIR" >&2
    exit 1
fi
CURRENT_STAGING_DIR_ABS=$(cd "$CURRENT_STAGING_DIR" && pwd)

# Determine privacy flag (only add --private if needed, default is public)
PRIVATE_FLAG=""
if [ "$PRIVATE" = true ]; then
    PRIVATE_FLAG="--private"
fi

echo "Creating space: $SPACE_REPO"
echo "Command: hf repo create $SPACE_REPO --repo-type space --space-sdk docker --exist-ok $PRIVATE_FLAG ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}"
# create the space if it doesn't exist
# Temporarily disable exit-on-error for this command
set +e
CREATE_OUTPUT=$(hf repo create "$SPACE_REPO" --repo-type space --space-sdk docker --exist-ok $PRIVATE_FLAG ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} 2>&1)
CREATE_EXIT_CODE=$?
set -e
if [ $CREATE_EXIT_CODE -ne 0 ]; then
    echo "❌ Space creation failed with exit code $CREATE_EXIT_CODE" >&2
    echo "Error output:" >&2
    echo "$CREATE_OUTPUT" >&2
    echo "" >&2
fi

echo "Uploading files to space: $SPACE_REPO"
echo "Command: hf upload --repo-type=space $PRIVATE_FLAG ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} $SPACE_REPO $CURRENT_STAGING_DIR_ABS"
# upload the staged content (if repo doesn't exist, it will be created with the privacy setting)
SPACE_UPLOAD_RESULT=$(hf upload --repo-type=space $PRIVATE_FLAG ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} "$SPACE_REPO" "$CURRENT_STAGING_DIR_ABS" 2>&1)
UPLOAD_EXIT_CODE=$?
if [ $UPLOAD_EXIT_CODE -ne 0 ]; then
    echo "❌ Upload failed with exit code $UPLOAD_EXIT_CODE" >&2
    echo "Error output:" >&2
    echo "$SPACE_UPLOAD_RESULT" >&2
    echo "" >&2
    echo "  Space: $SPACE_REPO" >&2
    echo "  Staging dir: $CURRENT_STAGING_DIR_ABS" >&2
    echo "  Files to upload:" >&2
    ls -la "$CURRENT_STAGING_DIR_ABS" >&2 || true
    exit 1
fi
# print the URL of the deployed space
echo "✅ Upload completed for https://huggingface.co/spaces/$SPACE_REPO"

# Cleanup the staging directory after successful deployment
if [ -d "$CURRENT_STAGING_DIR_ABS" ]; then
    rm -rf "$CURRENT_STAGING_DIR_ABS"
fi
