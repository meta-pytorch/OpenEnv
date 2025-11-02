# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Builder module for preparing environments for deployment."""

import os
import re
import shutil
from pathlib import Path
from typing import Optional


def prepare_staging_directory(env_name: str, base_image: str, staging_root: str = "hf-staging") -> Path:
    """
    Prepare staging directory structure for deployment.
    
    Args:
        env_name: Name of the environment.
        base_image: Base Docker image to use.
        staging_root: Root directory for staging (default: "hf-staging").
        
    Returns:
        Path to the staging directory.
    """
    # Create staging directory
    staging_dir = Path(staging_root) / env_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    
    # Create subdirectories
    (staging_dir / "src" / "core").mkdir(parents=True)
    (staging_dir / "src" / "envs" / env_name).mkdir(parents=True)
    
    return staging_dir


def copy_environment_files(env_name: str, staging_dir: Path) -> None:
    """
    Copy environment files to staging directory.
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
    """
    # Copy core files
    core_src = Path("src/core")
    core_dst = staging_dir / "src" / "core"
    if core_src.exists():
        shutil.copytree(core_src, core_dst, dirs_exist_ok=True)
    
    # Copy environment files
    env_src = Path("src/envs") / env_name
    env_dst = staging_dir / "src" / "envs" / env_name
    if not env_src.exists():
        raise FileNotFoundError(f"Environment not found: {env_src}")
    shutil.copytree(env_src, env_dst, dirs_exist_ok=True)


def prepare_dockerfile(env_name: str, staging_dir: Path, base_image: str) -> None:
    """
    Prepare Dockerfile for deployment.
    
    Uses the environment's Dockerfile if it exists, otherwise creates a default one.
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
        base_image: Base Docker image to use.
    """
    env_dockerfile = Path("src/envs") / env_name / "server" / "Dockerfile"
    dockerfile_path = staging_dir / "Dockerfile"
    
    if env_dockerfile.exists():
        # Copy and modify existing Dockerfile
        content = env_dockerfile.read_text()
        
        # Replace BASE_IMAGE references
        content = re.sub(
            r'ARG\s+BASE_IMAGE=.*',
            f'ARG BASE_IMAGE={base_image}',
            content
        )
        content = re.sub(
            r'FROM\s+\$\{BASE_IMAGE\}',
            f'FROM {base_image}',
            content
        )
        # Also handle direct FROM statements
        content = re.sub(
            r'FROM\s+openenv-base:.*',
            f'FROM {base_image}',
            content
        )
        
        dockerfile_path.write_text(content)
    else:
        # Create default Dockerfile
        default_dockerfile = f"""# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the specified openenv-base image
FROM {base_image}

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/{env_name}/ /app/src/envs/{env_name}/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.{env_name}.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        dockerfile_path.write_text(default_dockerfile)
    
    # Ensure web interface is enabled
    content = dockerfile_path.read_text()
    if "ENABLE_WEB_INTERFACE" not in content:
        # Add before the CMD line
        content = re.sub(
            r'(CMD\s+\[)',
            r'ENV ENABLE_WEB_INTERFACE=true\n\n\1',
            content
        )
        dockerfile_path.write_text(content)


def prepare_readme(env_name: str, staging_dir: Path) -> None:
    """
    Prepare README.md with HuggingFace front matter.
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
    """
    # Capitalize first letter of environment name
    env_title = env_name[0].upper() + env_name[1:] if env_name else env_name
    
    # Set environment-specific colors and emoji
    emoji_map = {
        "atari_env": ("🕹️", "red", "yellow"),
        "coding_env": ("💻", "blue", "gray"),
        "openspiel_env": ("🎮", "purple", "indigo"),
        "echo_env": ("🔊", "blue", "gray"),
        "chat_env": ("💬", "blue", "green"),
        "textarena_env": ("📜", "green", "blue"),
    }
    
    emoji, color_from, color_to = emoji_map.get(env_name, ("🐳", "blue", "green"))
    
    # Create README with front matter
    readme_content = f"""---
title: {env_title} Environment Server
emoji: {emoji}
colorFrom: {color_from}
colorTo: {color_to}
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# {env_title} Environment Server

FastAPI server for {env_name} environment powered by Meta's OpenEnv.

## About

This Space provides a containerized environment for {env_name} interactions.
Built with FastAPI and OpenEnv framework.

## Web Interface

This deployment includes an interactive web interface for exploring the environment:
- **HumanAgent Interface**: Interact with the environment using a web form
- **State Observer**: Real-time view of environment state and action history
- **Live Updates**: WebSocket-based real-time updates

Access the web interface at: `/web`

## API Documentation

Visit `/docs` for interactive API documentation.

## Health Check

The environment provides a health check endpoint at `/health`.
"""
    
    # Try to append content from original README if it exists
    original_readme = Path("src/envs") / env_name / "README.md"
    if original_readme.exists():
        original_content = original_readme.read_text()
        
        # Skip front matter if present
        if original_content.startswith("---"):
            # Find closing ---
            lines = original_content.split("\n")
            closing_idx = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    closing_idx = i
                    break
            
            if closing_idx:
                # Skip front matter
                original_content = "\n".join(lines[closing_idx + 1:])
        
        # Append original content
        readme_content += "\n" + original_content
    
    readme_path = staging_dir / "README.md"
    readme_path.write_text(readme_content)
