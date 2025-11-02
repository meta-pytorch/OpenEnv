# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment loader utilities."""

from pathlib import Path
from typing import Dict, Any, Optional


def validate_environment(env_name: str) -> Path:
    """
    Validate that environment exists and return its path.
    
    Args:
        env_name: Name of the environment to validate.
        
    Returns:
        Path to the environment directory.
        
    Raises:
        FileNotFoundError: If environment does not exist.
    """
    env_path = Path("src/envs") / env_name
    if not env_path.exists():
        raise FileNotFoundError(
            f"Environment '{env_name}' not found under src/envs. "
            f"Expected path: {env_path.absolute()}"
        )
    if not env_path.is_dir():
        raise FileNotFoundError(
            f"Environment '{env_name}' is not a directory. "
            f"Path: {env_path.absolute()}"
        )
    return env_path


def load_env_metadata(env_name: str) -> Dict[str, Any]:
    """
    Load environment metadata.
    
    Args:
        env_name: Name of the environment.
        
    Returns:
        Dictionary with environment metadata.
    """
    env_path = validate_environment(env_name)
    
    metadata: Dict[str, Any] = {
        "name": env_name,
        "path": str(env_path),
    }
    
    # Load README if it exists
    readme_path = env_path / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()
        metadata["readme"] = readme_content
        
        # Try to extract title from README
        lines = readme_content.split("\n")
        for line in lines:
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break
    
    # Check for server directory
    server_path = env_path / "server"
    if server_path.exists():
        metadata["has_server"] = True
        
        # Check for Dockerfile
        dockerfile_path = server_path / "Dockerfile"
        if dockerfile_path.exists():
            metadata["has_dockerfile"] = True
            metadata["dockerfile_path"] = str(dockerfile_path)
    
    # Check for models.py
    models_path = env_path / "models.py"
    if models_path.exists():
        metadata["has_models"] = True
    
    # Check for client.py
    client_path = env_path / "client.py"
    if client_path.exists():
        metadata["has_client"] = True
    
    return metadata
