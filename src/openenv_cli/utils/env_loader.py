# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment loader utilities."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .manifest import load_manifest


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


def validate_environment_at(env_root: Path) -> Path:
    """
    Validate that a given path is an environment root and return its path.
    
    An environment root is a directory that typically contains environment files, e.g.:
    - README.md
    - models.py
    - client.py
    - server/ (with Dockerfile)
    
    Args:
        env_root: Path to the environment root directory.
    
    Returns:
        Path to the environment directory.
    
    Raises:
        FileNotFoundError: If env_root does not exist or is not a directory.
    """
    if not env_root.exists():
        raise FileNotFoundError(
            f"Environment directory not found. Expected path: {env_root.absolute()}"
        )
    if not env_root.is_dir():
        raise FileNotFoundError(
            f"Environment path is not a directory. Path: {env_root.absolute()}"
        )
    # Require minimal environment structure: a 'server' directory
    server_dir = env_root / "server"
    if not server_dir.exists() or not server_dir.is_dir():
        raise FileNotFoundError(
            "Not a valid environment root. Expected a directory containing 'server/'. "
            "Run this command from the environment root (e.g., src/envs/<env_name>) or pass --env-path to it."
        )
    return env_root


def resolve_environment(env_name: Optional[str] = None, env_path: Optional[str] = None) -> Tuple[str, Path]:
    """
    Resolve environment name and root directory from either an explicit path,
    the current working directory, or the repo structure.
    
    Priority:
    1) If env_path is provided, use it as env root (env_name defaults to directory name if None)
    2) If env_name provided and src/envs/<env_name> exists, use that
    3) Otherwise, assume current working directory is the environment root (env_name = cwd name)
    """
    if env_path is not None:
        root = Path(env_path).resolve()
        validate_environment_at(root)
        # Prefer manifest name if present
        man = load_manifest(root)
        name = (man.name if man and man.name else (env_name if env_name is not None else root.name))
        return name, root

    if env_name is not None:
        # Try repo structure
        repo_env = (Path("src/envs") / env_name).resolve()
        if repo_env.exists() and repo_env.is_dir():
            return env_name, repo_env

    # Fallback: assume cwd is env root
    cwd = Path.cwd().resolve()
    validate_environment_at(cwd)
    man = load_manifest(cwd)
    name = (man.name if man and man.name else (env_name if env_name is not None else cwd.name))
    return name, cwd


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
