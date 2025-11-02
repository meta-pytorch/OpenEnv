#!/bin/bash

# OpenEnv Hugging Face Deployment Script for PRs
# This script deploys a new environment to Hugging Face Spaces

set -e

# Parse command line arguments
ENV_NAME=""
SPACE_SUFFIX=""
HUB_TAG="openenv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV_NAME="$2"
            shift 2
            ;;
        --space-suffix)
            SPACE_SUFFIX="$2"
            shift 2
            ;;
        --hub-tag)
            HUB_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$ENV_NAME" ]; then
    echo "Error: --env parameter is required"
    echo "Usage: $0 --env <environment_name> [--space-suffix <suffix>] [--hub-tag <tag>]"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    exit 1
fi

if [ -z "$HF_USERNAME" ]; then
    HF_USERNAME="openenv"
    echo "Using default HF_USERNAME: $HF_USERNAME"
fi

# Set space name
SPACE_NAME="${ENV_NAME}${SPACE_SUFFIX}"

echo "=========================================="
echo "Deploying $ENV_NAME to Hugging Face Space"
echo "Space name: $SPACE_NAME"
echo "Hub tag: $HUB_TAG"
echo "=========================================="

# Step 1: Prepare files using existing script
echo "Step 1: Preparing files for deployment..."
chmod +x scripts/prepare_hf_deployment.sh
./scripts/prepare_hf_deployment.sh "$ENV_NAME" ""

STAGING_DIR="hf-staging_${ENV_NAME}"

if [ ! -d "$STAGING_DIR" ]; then
    echo "Error: Staging directory $STAGING_DIR not found"
    exit 1
fi

# Step 2: Clone or create HF Space
echo "Step 2: Setting up Hugging Face Space..."
HF_SPACE_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
HF_SPACE_DIR="hf-space-${ENV_NAME}"

rm -rf "$HF_SPACE_DIR"

if git clone "$HF_SPACE_URL" "$HF_SPACE_DIR" 2>/dev/null; then
    echo "✓ Space exists, updating..."
else
    echo "✓ Creating new space..."
    mkdir -p "$HF_SPACE_DIR"
    cd "$HF_SPACE_DIR"
    git init
    git remote add origin "$HF_SPACE_URL"
    cd ..
fi

# Step 3: Copy prepared files
echo "Step 3: Copying files to space..."
cd "$HF_SPACE_DIR"

# Clear existing files (except .git)
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Copy all prepared files
cp -r "../${STAGING_DIR}"/* .

# Create README.md if it doesn't exist
if [ ! -f "README.md" ]; then
    cat > README.md << 'EOF'
---
title: OpenEnv Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# OpenEnv Environment

This is an environment from the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Usage

This space provides an HTTP API for interacting with the environment. See the OpenEnv documentation for details.
EOF
fi

# Step 4: Commit and push
echo "Step 4: Deploying to Hugging Face..."

git config user.email "github-actions[bot]@users.noreply.github.com"
git config user.name "github-actions[bot]"

if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "🤖 Deploy $ENV_NAME environment - $(date +'%Y-%m-%d %H:%M:%S')

Deployed from PR
Environment: $ENV_NAME
Tag: $HUB_TAG"

    echo "Pushing to Hugging Face..."

    # Try to push to main first, then master as fallback
    if git push origin main 2>/dev/null; then
        echo "✅ Successfully deployed to https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    elif git push origin master 2>/dev/null; then
        echo "✅ Successfully deployed to https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
    else
        echo "❌ Failed to push to Hugging Face"
        cd ..
        rm -rf "$HF_SPACE_DIR"
        rm -rf "$STAGING_DIR"
        exit 1
    fi
else
    echo "ℹ️  No changes to deploy"
fi

# Step 5: Cleanup
cd ..
echo "Step 5: Cleaning up..."
rm -rf "$HF_SPACE_DIR"
rm -rf "$STAGING_DIR"

echo "=========================================="
echo "✅ Deployment complete!"
echo "Space URL: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
echo "=========================================="
