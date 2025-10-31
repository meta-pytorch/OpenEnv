"""HuggingFace API helpers for OpenEnv CLI."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from huggingface_hub import HfApi, login as hf_login, upload_folder
from huggingface_hub.utils import HfHubHTTPError


# Hardcoded OpenEnv Environment Hub collection ID
OPENENV_COLLECTION_ID = "openenv/environment-hub-68f16377abea1ea114fa0743"


def ensure_authenticated(api: Optional[HfApi] = None) -> HfApi:
    """
    Ensure user is authenticated with HuggingFace.
    
    Returns an authenticated HfApi instance.
    Raises SystemExit if authentication fails.
    """
    if api is None:
        api = HfApi()
    
    # Try to get current user
    try:
        user = api.whoami()
        return api
    except Exception:
        # Not authenticated, try to login
        pass
    
    # Try using HUGGINGFACE_TOKEN environment variable
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        try:
            hf_login(token=token)
            api = HfApi(token=token)
            user = api.whoami()
            return api
        except Exception as e:
            print(f"Failed to authenticate with HUGGINGFACE_TOKEN: {e}", file=sys.stderr)
    
    # Fall back to huggingface-cli login
    print("Not authenticated with HuggingFace. Attempting login...", file=sys.stderr)
    if not sys.stdin.isatty():
        print(
            "ERROR: No TTY available for interactive login. "
            "Please set HUGGINGFACE_TOKEN environment variable or run 'huggingface-cli login' manually.",
            file=sys.stderr
        )
        sys.exit(1)
    
    try:
        result = subprocess.run(
            ["huggingface-cli", "login"],
            check=True,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        # Recheck after login
        api = HfApi()
        user = api.whoami()
        return api
    except subprocess.CalledProcessError:
        print(
            "ERROR: Failed to login via huggingface-cli. "
            "Please set HUGGINGFACE_TOKEN environment variable or run 'huggingface-cli login' manually.",
            file=sys.stderr
        )
        sys.exit(1)
    except FileNotFoundError:
        print(
            "ERROR: huggingface-cli not found. "
            "Please install it with: pip install -U 'huggingface_hub[cli]' "
            "or set HUGGINGFACE_TOKEN environment variable.",
            file=sys.stderr
        )
        sys.exit(1)


def ensure_space(
    api: HfApi,
    space_id: str,
    private: bool = False,
    hardware: Optional[str] = None,
) -> None:
    """
    Ensure a HuggingFace Space exists, creating it if necessary.
    
    Args:
        api: Authenticated HfApi instance
        space_id: Space ID in format "owner/space_name"
        private: Whether the space should be private
        hardware: Optional hardware type (e.g., "cpu-small", "t4-small")
    """
    owner, space_name = space_id.split("/", 1)
    
    # Check if space exists
    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        print(f"Space {space_id} already exists")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            # Space doesn't exist, create it
            print(f"Creating Space {space_id}...")
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="docker",
                private=private,
            )
            print(f"✅ Created Space {space_id}")
        else:
            raise
    
    # Request hardware if specified
    if hardware:
        try:
            api.request_space_hardware(repo_id=space_id, hardware=hardware)
            print(f"Requested hardware: {hardware}")
        except Exception as e:
            print(f"Warning: Failed to request hardware {hardware}: {e}", file=sys.stderr)


def upload_to_space(
    api: HfApi,
    space_id: str,
    local_path: Path,
    commit_message: Optional[str] = None,
) -> None:
    """
    Upload files to a HuggingFace Space.
    
    Args:
        api: Authenticated HfApi instance
        space_id: Space ID in format "owner/space_name"
        local_path: Local directory to upload
        commit_message: Optional commit message
    """
    if commit_message is None:
        commit_message = f"Deploy environment via OpenEnv CLI"
    
    print(f"Uploading files to {space_id}...")
    upload_folder(
        folder_path=str(local_path),
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_message,
        ignore_patterns=[".git", "__pycache__", "*.pyc"],
    )
    print(f"✅ Uploaded files to {space_id}")


def wait_for_space_build(api: HfApi, space_id: str, timeout: int = 600) -> None:
    """
    Wait for a Space to finish building.
    
    Args:
        api: Authenticated HfApi instance
        space_id: Space ID in format "owner/space_name"
        timeout: Maximum time to wait in seconds (default: 10 minutes)
    """
    print(f"Waiting for Space {space_id} to build...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            runtime = api.get_space_runtime(repo_id=space_id)
            stage = runtime.get("stage", "BUILDING")
            
            if stage == "RUNNING":
                print(f"✅ Space {space_id} is running!")
                return
            elif stage == "BUILDING":
                print(f"⏳ Space is building... (stage: {stage})")
            elif stage in ["NO_APP_FILE", "CONFIG_ERROR", "BUILD_ERROR"]:
                print(
                    f"❌ Space build failed with stage: {stage}. "
                    f"Check logs at https://huggingface.co/spaces/{space_id}/settings",
                    file=sys.stderr
                )
                sys.exit(1)
            else:
                print(f"⏳ Space status: {stage}")
        except Exception as e:
            print(f"Warning: Error checking space status: {e}", file=sys.stderr)
        
        time.sleep(5)
    
    print(
        f"⚠️  Timeout waiting for Space to build. "
        f"Check status at https://huggingface.co/spaces/{space_id}",
        file=sys.stderr
    )


def add_to_collection(api: HfApi, space_id: str, collection_id: str = OPENENV_COLLECTION_ID) -> None:
    """
    Add a Space to a HuggingFace collection.
    
    Args:
        api: Authenticated HfApi instance
        space_id: Space ID in format "owner/space_name"
        collection_id: Collection ID (defaults to OpenEnv Environment Hub)
    """
    print(f"Adding {space_id} to collection {collection_id}...")
    
    # Use the collections API endpoint
    # The API endpoint is: POST /api/collections/{collection_id}/items
    try:
        # We'll use the internal API client to make the request
        # This is a workaround since huggingface_hub may not have direct collection support
        token = api.token
        if not token:
            print("Warning: No token available for collection API", file=sys.stderr)
            return
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        # Collection API endpoint
        # Collection ID format: "owner/collection-name" - use as-is in URL
        url = f"https://huggingface.co/api/collections/{collection_id}/items"
        
        payload = {
            "item_id": space_id,
            "item_type": "space",
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Added {space_id} to collection {collection_id}")
        elif response.status_code == 409:
            print(f"ℹ️  {space_id} is already in collection {collection_id}")
        else:
            print(
                f"Warning: Failed to add to collection (status {response.status_code}): {response.text}. "
                f"Please add manually at https://huggingface.co/collections/{collection_id}",
                file=sys.stderr
            )
    except Exception as e:
        print(
            f"Warning: Failed to add to collection: {e}. "
            f"Please add manually at https://huggingface.co/collections/{collection_id}",
            file=sys.stderr
        )

