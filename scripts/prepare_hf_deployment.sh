#!/bin/bash

# OpenEnv Hugging Face Deployment Preparation Script
# This script prepares files for deployment to Hugging Face Spaces

set -e

# Cross-platform sed in-place editing
# BSD sed (macOS) requires -i '', GNU sed (Linux) requires -i
sed_inplace() {
    if sed --version >/dev/null 2>&1; then
        # GNU sed
        sed -i "$@"
    else
        # BSD sed
        sed -i '' "$@"
    fi
}

ENV_NAME="$1"
BASE_IMAGE_SHA="$2"
STAGING_DIR="hf-staging"

if [ -z "$ENV_NAME" ]; then
    echo "Error: Environment name is required"
    exit 1
fi

# Validate environment name
ENV_NAMES="$ENV_NAME"

# Set base image reference (using GHCR)
if [ -n "$BASE_IMAGE_SHA" ]; then
    BASE_IMAGE_REF="ghcr.io/meta-pytorch/openenv-base:$BASE_IMAGE_SHA"
    echo "Using specific SHA for openenv-base: $BASE_IMAGE_SHA"
else
    BASE_IMAGE_REF="ghcr.io/meta-pytorch/openenv-base:latest"
    echo "Using latest tag for openenv-base"
fi

echo "Preparing $ENV_NAME environment for deployment..."

# Create staging directory
CURRENT_STAGING_DIR="${STAGING_DIR}_${ENV_NAME}"
mkdir -p $CURRENT_STAGING_DIR/src/core
mkdir -p $CURRENT_STAGING_DIR/src/envs/$ENV_NAME

# Copy core files
cp -r src/core/* $CURRENT_STAGING_DIR/src/core/
echo "Copied core files"

# Copy environment files
cp -r src/envs/$ENV_NAME/* $CURRENT_STAGING_DIR/src/envs/$ENV_NAME/
echo "Copied $ENV_NAME environment files"

# Copy and modify the static Dockerfile from the environment
create_environment_dockerfile() {
    local env_name=$1
    local dockerfile_path="src/envs/$env_name/server/Dockerfile"
    local prepare_script="src/envs/$env_name/server/prepare_hf.sh"

    if [ ! -f "$dockerfile_path" ]; then
        echo "Error: Dockerfile not found at $dockerfile_path"
        exit 1
    fi

    # Copy the static Dockerfile
    cp "$dockerfile_path" "$CURRENT_STAGING_DIR/Dockerfile"
    echo "Copied static Dockerfile from $dockerfile_path"

    # Check if environment has custom HF preparation script
    if [ -f "$prepare_script" ]; then
        echo "Found custom HF preparation script, executing..."
        chmod +x "$prepare_script"
        "$prepare_script" "$CURRENT_STAGING_DIR/Dockerfile" "$BASE_IMAGE_REF"
    else
        # Standard Dockerfile modification: replace ARG BASE_IMAGE with FROM
        sed_inplace "s|ARG BASE_IMAGE=.*||g" "$CURRENT_STAGING_DIR/Dockerfile"
        sed_inplace "s|FROM \${BASE_IMAGE}|FROM $BASE_IMAGE_REF|g" "$CURRENT_STAGING_DIR/Dockerfile"
        echo "Modified Dockerfile with base image: $BASE_IMAGE_REF"
    fi

    # Add web interface support before the final CMD
    # Use awk for cross-platform compatibility
    awk '/^CMD \[/{print "ENV ENABLE_WEB_INTERFACE=true\n"; print; next} 1' "$CURRENT_STAGING_DIR/Dockerfile" > "$CURRENT_STAGING_DIR/Dockerfile.tmp"
    mv "$CURRENT_STAGING_DIR/Dockerfile.tmp" "$CURRENT_STAGING_DIR/Dockerfile"
    echo "Enabled web interface"
}

create_environment_dockerfile $ENV_NAME

# Copy and prepend HF-specific intro to README
create_readme() {
    local env_name=$1
    local readme_source="src/envs/$env_name/README.md"

    if [ ! -f "$readme_source" ]; then
        echo "Error: README not found at $readme_source"
        exit 1
    fi

    # Check if README already has HF front matter
    if head -n 1 "$readme_source" | grep -q "^---$"; then
        echo "README has HF front matter, inserting HF deployment section after it"

        # Find the line number of the closing --- (second occurrence)
        local closing_line=$(grep -n "^---$" "$readme_source" | sed -n '2p' | cut -d: -f1)

        if [ -z "$closing_line" ]; then
            echo "Error: Could not find closing --- in front matter"
            exit 1
        fi

        # Split the README: front matter + rest
        head -n "$closing_line" "$readme_source" > "$CURRENT_STAGING_DIR/README.md"

        # Add HF-specific deployment info right after front matter
        cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'

## 🚀 Hugging Face Space Deployment

This is a Hugging Face Space deployment of the OpenEnv environment. It includes:

- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring

### Connecting from Code

```python
from envs.ENV_NAME_PLACEHOLDER import ENV_CLASS_PLACEHOLDER

# Connect to this HF Space
env = ENV_CLASS_PLACEHOLDER(base_url="https://huggingface.co/spaces/openenv/ENV_NAME_PLACEHOLDER")

# Use the environment
result = env.reset()
result = env.step(action)
```

For full documentation, see the [OpenEnv repository](https://github.com/meta-pytorch/OpenEnv).

README_EOF

        # Append the rest of the original README (skip front matter)
        tail -n "+$((closing_line + 1))" "$readme_source" >> "$CURRENT_STAGING_DIR/README.md"
    else
        echo "Error: README missing HF front matter at $readme_source"
        echo "Please add YAML front matter to the environment README"
        exit 1
    fi

    # Set environment-specific class name
    case $env_name in
        "echo_env") ENV_CLASS="EchoEnv" ;;
        "coding_env") ENV_CLASS="CodingEnv" ;;
        "chat_env") ENV_CLASS="ChatEnv" ;;
        "atari_env") ENV_CLASS="AtariEnv" ;;
        "openspiel_env") ENV_CLASS="OpenSpielEnv" ;;
        "wildfire_env") ENV_CLASS="WildfireEnv" ;;
        *) ENV_CLASS="Env" ;;
    esac

    # Replace placeholders (cross-platform)
    sed_inplace "s/ENV_NAME_PLACEHOLDER/$env_name/g" "$CURRENT_STAGING_DIR/README.md"
    sed_inplace "s/ENV_CLASS_PLACEHOLDER/$ENV_CLASS/g" "$CURRENT_STAGING_DIR/README.md"
}

create_readme $ENV_NAME
echo "Copied and enhanced README for HF Space"
echo "Completed preparation for $ENV_NAME environment"
