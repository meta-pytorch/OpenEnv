# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for OpenEnv."""

import argparse
import sys

from .commands.push import push_environment


def main():
    """Main entry point for OpenEnv CLI."""
    parser = argparse.ArgumentParser(
        prog="openenv",
        description="OpenEnv CLI - Manage and deploy OpenEnv environments",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Push command
    push_parser = subparsers.add_parser(
        "push",
        help="Push an environment to HuggingFace Spaces",
    )
    push_parser.add_argument(
        "env_name",
        help="Name of the environment to push (e.g., echo_env)",
    )
    push_parser.add_argument(
        "--namespace",
        help="HuggingFace namespace (organization or user). "
             "If not provided, uses authenticated user's username.",
    )
    push_parser.add_argument(
        "--space-name",
        help="Custom name for the HuggingFace Space. "
             "If not provided, uses the environment name.",
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private space (default: public)",
    )
    push_parser.add_argument(
        "--base-image",
        help="Base Docker image to use "
             "(default: ghcr.io/meta-pytorch/openenv-base:latest)",
    )
    push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but don't upload to HuggingFace",
    )
    
    args = parser.parse_args()
    
    if args.command == "push":
        try:
            push_environment(
                env_name=args.env_name,
                namespace=args.namespace,
                space_name=args.space_name,
                private=args.private,
                base_image=args.base_image,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
