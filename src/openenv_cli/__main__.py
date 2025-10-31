"""OpenEnv CLI main entry point."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .hf import (
    add_to_collection,
    ensure_authenticated,
    ensure_space,
    upload_to_space,
    wait_for_space_build,
)


def validate_environment(env_name: str) -> Path:
    """
    Validate that the environment exists and has required files.
    
    Returns the path to the environment directory.
    Raises SystemExit if validation fails.
    """
    # __file__ is at src/openenv_cli/__main__.py
    # So: __file__ -> src/openenv_cli -> src -> project_root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / "src" / "envs" / env_name
    
    if not env_path.exists():
        print(
            f"ERROR: Environment '{env_name}' not found at {env_path}",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Check for required files
    dockerfile_path = env_path / "server" / "Dockerfile"
    readme_path = env_path / "README.md"
    
    missing = []
    if not dockerfile_path.exists():
        missing.append(f"server/Dockerfile at {dockerfile_path}")
    if not readme_path.exists():
        missing.append(f"README.md at {readme_path}")
    
    if missing:
        print(
            f"ERROR: Environment '{env_name}' is missing required files:\n"
            + "\n".join(f"  - {m}" for m in missing),
            file=sys.stderr
        )
        sys.exit(1)
    
    # Check README has HF front matter
    try:
        readme_content = readme_path.read_text()
        if not readme_content.startswith("---\n"):
            print(
                f"ERROR: README.md must have HuggingFace front matter (YAML between --- markers)",
                file=sys.stderr
            )
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read README.md: {e}", file=sys.stderr)
        sys.exit(1)
    
    return env_path


def stage_build(env_name: str, base_image_sha: Optional[str] = None) -> Path:
    """
    Stage the build for HuggingFace Space using prepare_hf_deployment.sh.
    
    Returns the path to the staging directory.
    """
    # __file__ is at src/openenv_cli/__main__.py
    # So: __file__ -> src/openenv_cli -> src -> project_root
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "scripts" / "prepare_hf_deployment.sh"
    staging_dir = project_root / f"hf-staging_{env_name}"
    
    if not script_path.exists():
        print(f"ERROR: Deployment script not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    
    # Make script executable
    script_path.chmod(0o755)
    
    # Build command
    cmd = [str(script_path), env_name]
    if base_image_sha:
        cmd.append(base_image_sha)
    
    print(f"Staging build for {env_name}...")
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to stage build: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not staging_dir.exists():
        print(f"ERROR: Staging directory not created at {staging_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Staged build in {staging_dir}")
    return staging_dir


def push_command(args: argparse.Namespace) -> None:
    """Execute the push command."""
    # Validate environment
    env_path = validate_environment(args.env)
    
    # Authenticate
    api = ensure_authenticated()
    user_info = api.whoami()
    print(f"✅ Authenticated as {user_info.get('name', 'unknown')}")
    
    # Resolve space ID
    if args.space:
        space_id = args.space
    elif args.org:
        space_name = args.name or args.env
        space_id = f"{args.org}/{space_name}"
    else:
        # Default to user's personal namespace
        username = user_info.get("name", "unknown")
        space_id = f"{username}/{args.env}"
    
    print(f"Target Space: {space_id}")
    
    # Stage build
    staging_dir = stage_build(args.env, args.base_image_sha)
    
    # Provision Space
    ensure_space(api, space_id, private=args.private, hardware=args.hardware)
    
    # Upload deployment
    upload_to_space(api, space_id, staging_dir)
    
    # Wait for build if requested
    if args.wait:
        wait_for_space_build(api, space_id, timeout=args.timeout)
    
    # Add to collection (mandatory)
    add_to_collection(api, space_id)
    
    space_url = f"https://huggingface.co/spaces/{space_id}"
    print(f"\n✅ Environment deployed successfully!")
    print(f"🌐 Space URL: {space_url}")
    if args.wait:
        print(f"🚀 Your environment is ready to use!")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenEnv CLI - Manage environment deployments to HuggingFace Spaces",
        prog="openenv"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Push command
    push_parser = subparsers.add_parser(
        "push",
        help="Push an environment to HuggingFace Spaces",
        description="Deploy an environment to HuggingFace Spaces with web interface and collection integration"
    )
    
    push_parser.add_argument(
        "--env",
        required=True,
        help="Environment name (must exist in src/envs/{name})"
    )
    
    # Space targeting options (mutually exclusive group)
    space_group = push_parser.add_mutually_exclusive_group()
    space_group.add_argument(
        "--space",
        help="Full Space ID (format: owner/space_name)"
    )
    space_group.add_argument(
        "--org",
        help="Organization name for the Space"
    )
    
    push_parser.add_argument(
        "--name",
        help="Space name (defaults to environment name). Requires --org"
    )
    
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private Space"
    )
    
    push_parser.add_argument(
        "--hardware",
        help="Request specific hardware for the Space (e.g., cpu-basic, cpu-upgrade, t4-small, a10g-large)"
    )
    
    push_parser.add_argument(
        "--base-image-sha",
        help="Specific SHA/tag for the base image (defaults to 'latest')"
    )
    
    push_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for Space to finish building before exiting"
    )
    
    push_parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for --wait (default: 600)"
    )
    
    push_parser.set_defaults(func=push_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

