# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Builder module for preparing environments for deployment."""

import random
import re
import shutil
from pathlib import Path
import shutil as _shutil


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
    
    return staging_dir


def copy_environment_files(env_name: str, staging_dir: Path, env_root: Path | None = None) -> None:
    """
    Copy environment files to staging directory.
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
    """
    # Copy environment files from env_root or repo structure
    env_src = env_root if env_root is not None else (Path("src/envs") / env_name)
    if not env_src.exists():
        raise FileNotFoundError(f"Environment not found: {env_src}")
    # Ignore generated staging and caches to avoid recursive copies when running from env root
    ignore = _shutil.ignore_patterns("hf-staging", "__pycache__", ".git", "*.pyc", "*.pyo")
    shutil.copytree(env_src, staging_dir, dirs_exist_ok=True, ignore=ignore)
    
    # Also include openenv_core runtime so web interface is available in container
    core_pkg = Path("src/core/openenv_core")
    if core_pkg.exists():
        shutil.copytree(core_pkg, staging_dir / "openenv_core", dirs_exist_ok=True, ignore=ignore)


def prepare_dockerfile(env_name: str, staging_dir: Path, base_image: str, env_root: Path | None = None) -> None:
    """
    Prepare Dockerfile for deployment.
    
    Uses the environment's Dockerfile if it exists, otherwise creates a default one.
    Transforms template Dockerfiles (which use COPY . /app and server.app) to the
    flattened staging structure (COPY . /app).
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
        base_image: Base Docker image to use.
    """
    env_dockerfile = (env_root / "server" / "Dockerfile") if env_root is not None else (Path("src/envs") / env_name / "server" / "Dockerfile")
    dockerfile_path = staging_dir / "Dockerfile"
    reqs_exists = (staging_dir / "server" / "requirements.txt").exists()
    
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
        
        # Ensure a WORKDIR /app exists
        if not re.search(r'^WORKDIR\s+/app', content, flags=re.MULTILINE):
            content = re.sub(r'^(FROM\s+.*)$', r"\1\n\nWORKDIR /app", content, flags=re.MULTILINE)
        
        # Ensure COPY . /app (remove repo-structure COPYs if present)
        content = re.sub(r'^COPY\s+src/.*$', '', content, flags=re.MULTILINE)
        if not re.search(r'^COPY\s+\.\s+/app', content, flags=re.MULTILINE):
            content = re.sub(r'^(WORKDIR\s+/app.*)$', r"\1\n\nCOPY . /app", content, flags=re.MULTILINE)
        
        # If requirements exist, ensure they are installed after COPY
        if reqs_exists and "pip install" not in content:
            content = re.sub(
                r'^(COPY\s+\.\s+/app.*)$',
                r"\1\n\nRUN pip install --no-cache-dir -r server/requirements.txt",
                content,
                flags=re.MULTILINE
            )
        
        # Ensure uvicorn points to server.app:app (flat layout)
        content = re.sub(r'"envs\.[^"]*\.server\.app:app"', '"server.app:app"', content)
        content = re.sub(r'uvicorn\s+envs\.[^\s]*\.server\.app:app', 'uvicorn server.app:app', content)
        
        dockerfile_path.write_text(content)
    else:
        # Create default Dockerfile for flat layout
        install_line = "\nRUN pip install --no-cache-dir -r server/requirements.txt\n" if reqs_exists else "\n"
        default_dockerfile = f"""# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the specified openenv-base image
FROM {base_image}

WORKDIR /app

# Copy environment root contents
COPY . /app{install_line}
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        dockerfile_path.write_text(default_dockerfile)
    
    # Ensure web interface and Python path are enabled (must be before CMD)
    content = dockerfile_path.read_text()
    injections = []
    if "ENABLE_WEB_INTERFACE" not in content:
        injections.append("ENV ENABLE_WEB_INTERFACE=true")
    if "PYTHONPATH=/app" not in content:
        injections.append("ENV PYTHONPATH=/app")
    if injections:
        # Insert both lines before the CMD line (in order)
        content = re.sub(
            r'^(CMD\s+\[)',
            "\n".join(injections) + "\n\n" + r"\1",
            content,
            flags=re.MULTILINE
        )
        dockerfile_path.write_text(content)


def prepare_readme(env_name: str, staging_dir: Path, env_root: Path | None = None) -> None:
    """
    Prepare README.md with Hugging Face front matter.
    
    If the environment README already has Hugging Face front matter (starts and ends with `---`),
    use it as-is. Otherwise, generate front matter with random emoji and colors.
    
    Args:
        env_name: Name of the environment.
        staging_dir: Staging directory path.
    """
    # Check both src/envs/${ENV_NAME}/README.md and src/envs/${ENV_NAME}/server/README.md
    base = env_root if env_root is not None else (Path("src/envs") / env_name)
    readme_paths = [
        base / "README.md",
        base / "server" / "README.md",
    ]
    
    # Check if any README has Hugging Face front matter
    existing_readme_content = None
    for readme_path in readme_paths:
        if readme_path.exists():
            content = readme_path.read_text()
            # Check if it has front matter (starts with --- and has closing ---)
            if content.startswith("---"):
                lines = content.split("\n")
                # Look for closing --- (must be at the top, within first 100 lines)
                for i in range(1, min(100, len(lines))):
                    if lines[i].strip() == "---":
                        # Found front matter, use this README as-is
                        existing_readme_content = content
                        break
                if existing_readme_content:
                    break
    
    # If we found an existing README with front matter, use it as-is
    if existing_readme_content:
        readme_path = staging_dir / "README.md"
        readme_path.write_text(existing_readme_content)
        return
    
    # Otherwise, generate front matter with random emoji and colors
    # Safely capitalize env_name - handle empty string and None cases
    if env_name and len(env_name) > 0:
        env_title = env_name[0].upper() + env_name[1:]
    else:
        env_title = ""
    
    # Approved emojis from Spaces Configuration Reference
    approved_emojis = [
        "🎮", "🚀", "💻", "🔬", "🧪", "🎯", "🎨", "📊", "🤖", "🌟",
        "⚡", "🔧", "📱", "💡", "🎲", "🎵", "🎸", "🎭", "🎬", "🏆",
        "🔥", "💎", "🌈", "🎁", "🎈", "🎊", "🎉", "🦄", "🐳", "🐙",
        "🦋", "🐝", "🐞", "🦅", "🦉", "🦇", "🐉", "🦖", "🦕", "🐢",
        "🐍", "🦎", "🐊", "🐋", "🦈", "🐬", "🐠", "🐟", "🐡", "🦑",
        "🦐", "🦞", "🦀", "🐺", "🐗", "🐴", "🐛", "🐌", "🐜", "🦟",
        "🦗", "🕷️", "🦂", "🐅", "🐆", "🦓", "🦍", "🦧", "🐘", "🦛",
        "🦏", "🐪", "🐫", "🦒", "🦘", "🦬", "🐃", "🐂", "🐄", "🐎",
        "🐖", "🐏", "🐑", "🦙", "🐐", "🦌", "🐕", "🐩", "🦮", "🐈",
        "🪶", "🐓", "🦃", "🦚", "🦜", "🦢", "🦩", "🕊️", "🐇", "🐿️",
        "🦫", "🦔", "🦝", "🦨", "🦡", "🦦", "🦥", "🐁", "🐀", "🐾",
        "🐲", "🌵", "🎄", "🌲", "🌳", "🌴", "🌱", "🌿", "☘️", "🍀",
        "🎍", "🪴", "🎋", "🍃", "🍂", "🍁", "🍄", "🐚", "🪨", "🌾",
        "💐", "🌷", "🌹", "🥀", "🌺", "🌸", "🌼", "🌻", "🌞", "🌝",
        "🌛", "🌜", "🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔",
        "🌙", "🌚", "🌏", "🪐", "💫", "⭐", "✨", "☄️", "💥", "☀️",
        "⛅", "☁️", "⛈️", "🌤️", "🌦️", "🌧️", "🌩️", "❄️", "☃️", "⛄",
        "🌬️", "💨", "💧", "💦", "☔", "☂️", "🌊", "🌫️",
    ]
    
    # Approved colors from Spaces Configuration Reference
    approved_colors = ["red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"]
    
    # Randomly select emoji and colors
    emoji = random.choice(approved_emojis)
    color_from = random.choice(approved_colors)
    color_to = random.choice(approved_colors)
    
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
    
    # Try to append content from original README if it exists (without front matter)
    # Only append if the original README doesn't have front matter (if it has front matter,
    # we've already used it as-is above, so we shouldn't append here to avoid duplicates)
    original_readme = (env_root / "README.md") if env_root is not None else (Path("src/envs") / env_name / "README.md")
    if original_readme.exists():
        original_content = original_readme.read_text()
        
        # Only append if original README doesn't have front matter
        # (if it has front matter, we already used it as-is earlier, so skip to avoid duplicates)
        if not original_content.startswith("---"):
            # Original README has no front matter, append its content after generated sections
            # This preserves environment-specific documentation
            original_content_stripped = original_content.strip()
            if original_content_stripped:
                readme_content += "\n\n" + original_content_stripped
    
    readme_path = staging_dir / "README.md"
    readme_path.write_text(readme_content)
