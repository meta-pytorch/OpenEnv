# Converting Your Environment to OpenEnv Standard

This guide helps you convert an existing `src/envs/<env_name>` environment to a standalone, OpenEnv CLI-compatible environment that can be independently developed, versioned, and deployed.

## Overview

The new OpenEnv standard enables:
- **Independent repositories**: Each environment can have its own git repo
- **Standalone deployment**: Use `openenv push` to deploy directly to HuggingFace or Docker registries
- **Better dependency management**: Use `pyproject.toml` with `uv` for fast, reliable builds
- **Development flexibility**: Work on environments without the full OpenEnv monorepo

## Prerequisites

- Python 3.10+
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
- Docker (for building and testing)
- Git (for version control)

## Quick Start: Automated Conversion

We provide a script to automate most of the conversion process:

```bash
# From the OpenEnv repository root
./scripts/convert_env.sh src/envs/my_env /path/to/new/my_env_standalone
```

> **Note:** The converter requires `python3` on your PATH and works with the default Bash shipped on macOS. When prompted, answer `y` to proceed and leave the optional naming prompts blank to accept the defaults.

This script will:
1. Copy your environment to a new directory
2. Convert `requirements.txt` to `pyproject.toml` (if needed)
3. Add HuggingFace frontmatter to README
4. Update Dockerfile for standalone builds
5. Initialize a new git repository
6. Create necessary configuration files
7. Rewrite imports so the environment depends on `openenv-core` and installs as a proper Python package

After running the script, jump to [Step 4: Testing Your Conversion](#step-4-testing-your-conversion).

## Manual Conversion Steps

If you prefer to convert manually or need to understand what's happening:

### Step 1: Create New Environment Directory

```bash
# Create a new directory for your standalone environment
mkdir -p ~/my_projects/my_env_standalone
cd ~/my_projects/my_env_standalone

# Copy your existing environment
cp -r /path/to/OpenEnv/src/envs/my_env/* .

# Initialize git repository
git init
git add .
git commit -m "Initial commit: Convert from OpenEnv monorepo"
```

### Step 2: Convert Dependencies to pyproject.toml

If your environment uses `server/requirements.txt`, convert it to `pyproject.toml`:

#### Option A: Automated Conversion

```python
# Run this Python script in your environment directory
import re
from pathlib import Path

env_name = Path.cwd().name
requirements_file = Path("server/requirements.txt")

if requirements_file.exists():
    # Parse requirements.txt
    deps = []
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                deps.append(f'    "{line}",')
    
    deps_str = "\n".join(deps)
    
    # Create pyproject.toml
    pyproject_content = f'''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-{env_name}"
version = "0.1.0"
description = "{env_name.replace('_', ' ').title()} Environment for OpenEnv"
requires-python = ">=3.10"
dependencies = [
{deps_str}
    "openenv-core>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "ipykernel>=6.29.5",
]

[project.scripts]
server = "{env_name}.server.app:main"

[tool.setuptools]
packages = ["{env_name}", "{env_name}.server"]
package-dir = { "{env_name}" = ".", "{env_name}.server" = "server" }

[tool.setuptools.package-data]
{env_name} = ["**/*.yaml", "**/*.yml"]
'''
    
    Path("pyproject.toml").write_text(pyproject_content)
    print(f"✓ Created pyproject.toml from {requirements_file}")
else:
    print("No requirements.txt found - pyproject.toml may already exist")
```

#### Option B: Manual Creation

Create `pyproject.toml` at the environment root:

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-my_env"
version = "0.1.0"
description = "My Environment for OpenEnv"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.1.0",
    "fastapi>=0.115.0",
    "pydantic>=2.0.0",
    "uvicorn>=0.24.0",
    "requests>=2.25.0",
    # Add your environment-specific dependencies here
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "ipykernel>=6.29.5",
]

[project.scripts]
server = "my_env.server.app:main"

[tool.setuptools]
packages = ["{env_name}", "{env_name}.server"]
package-dir = { "{env_name}" = ".", "{env_name}.server" = "server" }
```

**Important**: Replace `my_env` with your actual environment name throughout the file.

### Step 3: Update Files for Standalone Usage

#### 3.1: Add HuggingFace Frontmatter to README.md

Add this YAML frontmatter to the top of your `README.md`:

```markdown
---
title: My Environment Server
emoji: 🎮
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# My Environment

[Rest of your README content...]
```

**Customize**:
- `title`: Your environment's display name
- `emoji`: Pick an emoji that represents your environment ([emoji list](https://emojipedia.org/))
- `colorFrom` / `colorTo`: Gradient colors for HuggingFace card (hex codes)

#### 3.2: Update Dockerfile

Your `server/Dockerfile` needs to support standalone builds. Update it to:

```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Run the server
CMD ["python", "-m", "uvicorn", "my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key changes**:
- Uses `pip install -e .` to install from `pyproject.toml`
- Copies entire environment directory
- Sets `ENABLE_WEB_INTERFACE=true` for HuggingFace deployments
- Replace `my_env` with your environment name

#### 3.3: Update app.py

Ensure your `server/app.py` has a proper `main()` function:

```python
# At the end of server/app.py
def main():
    """Main entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

This enables running with `uv run server`.

#### 3.4: Create uv.lock

Generate a lockfile for reproducible builds:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Generate lockfile
uv lock
```

This creates `uv.lock` which pins all dependencies.

#### 3.5: Add .gitignore

Create or update `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment outputs
outputs/
logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/
```

### Step 4: Testing Your Conversion

#### 4.1: Install and Test Locally

```bash
# Install in editable mode
uv pip install -e .

# Test imports
python -c "from my_env import MyAction, MyObservation, MyEnv"

# Run server locally
uv run server
```

Visit `http://localhost:8000/docs` to see the API documentation.

#### 4.2: Build Docker Image

```bash
# Build the image
openenv build

# Or manually:
docker build -t my_env:latest -f server/Dockerfile .

# Test the container
docker run -p 8000:8000 my_env:latest
```

#### 4.3: Validate Structure

Use the OpenEnv CLI to validate your environment:

```bash
openenv validate
```

This checks for:
- Required files (`openenv.yaml`, `pyproject.toml`, etc.)
- Correct dependency structure
- Valid server entry point
- Docker build capability

### Step 5: Deploy Your Environment

#### Deploy to HuggingFace Spaces

```bash
# Deploy to HuggingFace (with web interface)
openenv push

# Deploy to specific repo
openenv push --repo-id myusername/my-env

# Deploy privately
openenv push --private
```

#### Deploy to Docker Registry

```bash
# Build and push to Docker Hub
openenv push --registry docker.io/myusername

# Push to GitHub Container Registry
openenv push --registry ghcr.io/myorg

# Push to custom registry
openenv push --registry myregistry.io/path/to/repo
```

## Directory Structure After Conversion

Your standalone environment should look like this:

```
my_env_standalone/
├── .git/                  # Git repository
├── .gitignore            # Ignore patterns
├── README.md             # With HF frontmatter
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Dependencies and metadata
├── uv.lock              # Locked dependencies
├── __init__.py          # Module exports
├── client.py            # Environment client
├── models.py            # Action/Observation models
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI app (with main())
    ├── Dockerfile       # Standalone build
    └── my_env_environment.py  # Core environment logic
```

## Common Issues and Solutions

### Issue: Import Errors After Installation

**Solution**: Reinstall in editable mode:
```bash
uv pip install -e . --force-reinstall
```

### Issue: Docker Build Fails

**Solutions**:
1. Ensure `pyproject.toml` has all dependencies
2. Check Dockerfile COPY commands are correct
3. Verify base image is accessible

### Issue: `openenv` Commands Not Found

**Solution**: Install openenv-cli:
```bash
pip install openenv-cli
# or
uv pip install openenv-cli
```

### Issue: Server Entry Point Not Found

**Solution**: Ensure `pyproject.toml` has correct entry point:
```toml
[project.scripts]
server = "my_env.server.app:main"  # Replace my_env with your name
```

### Issue: Missing openenv-core Dependency

**Solution**: Add to `pyproject.toml`:
```toml
dependencies = [
    "openenv-core>=0.1.0",
    # ... other dependencies
]
```

For local development, install core from the OpenEnv repo:
```bash
pip install -e /path/to/OpenEnv/src/core
```

## Development Workflow

Once converted, you can work independently:

```bash
# Clone your standalone repo
git clone https://github.com/myusername/my_env_standalone
cd my_env_standalone

# Install dependencies
uv pip install -e .

# Run locally
uv run server

# Make changes to client.py, models.py, etc.

# Test changes
openenv validate
openenv build

# Deploy updates
openenv push
```
## Automated Script

For full automation, see the `scripts/convert_env.sh` script included in this repository.
