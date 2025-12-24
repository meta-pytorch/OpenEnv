#!/usr/bin/env python3
"""
Hugging Face Collection Manager for OpenEnv

This script automatically discovers Docker Spaces tagged with 'openenv' on Hugging Face
and adds them to the Environment Hub collection if they're not already present.

Usage:
    python scripts/manage_hf_collection.py [--dry-run] [--verbose]

Environment Variables:
    HF_TOKEN: Required. Your Hugging Face API token with write access to collections.
"""

import argparse
import logging
import os
import sys
from typing import Set, List
from huggingface_hub import HfApi, list_spaces
from huggingface_hub.utils import HfHubHTTPError


# Constants
COLLECTION_SLUG = "openenv/environment-hub-68f16377abea1ea114fa0743"
TAG_FILTER = "openenv"
SPACE_TYPE = "docker"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_api() -> HfApi:
    """
    Initialize and authenticate the Hugging Face API client.
    
    Returns:
        HfApi: Authenticated API client
        
    Raises:
        SystemExit: If HF_TOKEN is not set
    """
    hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        logger.error("HF_TOKEN environment variable is not set!")
        logger.error("Please set it with: export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    logger.info("Authenticating with Hugging Face...")
    api = HfApi(token=hf_token)
    
    try:
        whoami = api.whoami()
        logger.info(f"✓ Authenticated as: {whoami['name']}")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)
    
    return api


def get_collection_spaces(api: HfApi) -> Set[str]:
    """
    Retrieve the list of spaces currently in the Environment Hub collection.
    
    Args:
        api: Authenticated HfApi client
        
    Returns:
        Set of space IDs (in format "owner/space-name")
    """
    logger.info(f"Fetching current collection: {COLLECTION_SLUG}")
    
    try:
        collection = api.get_collection(COLLECTION_SLUG)
        
        # Extract space IDs from collection items
        space_ids = set()
        for item in collection.items:
            if item.item_type == "space":
                space_ids.add(item.item_id)
        
        logger.info(f"✓ Found {len(space_ids)} spaces in collection")
        return space_ids
        
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Collection not found: {COLLECTION_SLUG}")
            logger.error("Please ensure the collection exists and you have access to it")
        else:
            logger.error(f"Error fetching collection: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error fetching collection: {e}")
        sys.exit(1)


def discover_openenv_spaces(api: HfApi) -> List[str]:
    """
    Search for all Docker Spaces tagged with 'openenv'.
    
    Args:
        api: Authenticated HfApi client
        
    Returns:
        List of space IDs (in format "owner/space-name")
    """
    logger.info(f"Searching for Docker Spaces with tag '{TAG_FILTER}'...")
    
    try:
        # List all spaces with the openenv tag using search parameter
        spaces = list(list_spaces(
            search=TAG_FILTER,
            full=False,
            sort="trending_score",
            direction=-1
        ))
        
        # Filter for Docker spaces with the openenv tag
        # Note: search may return spaces that mention 'openenv' in description too,
        # so we need to verify the tag is actually present
        docker_spaces_with_tag = []
        for space in spaces:
            # Get full space info to check tags
            try:
                space_info = api.space_info(space.id)
                # Check if it's a Docker space and has the openenv tag
                if (hasattr(space_info, 'sdk') and space_info.sdk == 'docker' and
                    hasattr(space_info, 'tags') and TAG_FILTER in space_info.tags and 
                    space_info.runtime.stage != "RUNTIME_ERROR"):
                    docker_spaces_with_tag.append(space.id)
            except Exception as e:
                logger.warning(f"Could not fetch info for space {space.id}: {e}")
                continue
        
        logger.info(f"✓ Discovered {len(docker_spaces_with_tag)} Docker spaces with tag '{TAG_FILTER}'")
        
        return docker_spaces_with_tag
        
    except Exception as e:
        logger.error(f"Error discovering spaces: {e}")
        sys.exit(1)


def add_spaces_to_collection(
    api: HfApi,
    space_ids: List[str],
    dry_run: bool = False
) -> int:
    """
    Add new spaces to the Environment Hub collection.
    
    Args:
        api: Authenticated HfApi client
        space_ids: List of space IDs to add
        dry_run: If True, only simulate the addition without making changes
        
    Returns:
        Number of spaces added (or would be added in dry-run mode)
    """
    if not space_ids:
        logger.info("No new spaces to add")
        return 0
    
    added_count = 0
    failed_count = 0
    
    for space_id in space_ids:
        if dry_run:
            logger.info(f"[DRY RUN] Would add space: {space_id}")
            added_count += 1
        else:
            try:
                logger.info(f"Adding space to collection: {space_id}")
                api.add_collection_item(
                    collection_slug=COLLECTION_SLUG,
                    item_id=space_id,
                    item_type="space"
                )
                logger.info(f"✓ Successfully added: {space_id}")
                added_count += 1
            except HfHubHTTPError as e:
                if e.response.status_code == 409:
                    # Space already in collection (race condition)
                    logger.warning(f"Space already in collection: {space_id}")
                else:
                    logger.error(f"Failed to add {space_id}: {e}")
                    failed_count += 1
            except Exception as e:
                logger.error(f"Unexpected error adding {space_id}: {e}")
                failed_count += 1
    
    if failed_count > 0:
        logger.warning(f"Failed to add {failed_count} spaces")
    
    return added_count


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Manage Hugging Face Environment Hub collection for OpenEnv spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in dry-run mode to preview changes
  python scripts/manage_hf_collection.py --dry-run --verbose
  
  # Run for real to add spaces to collection
  python scripts/manage_hf_collection.py
  
  # View verbose output
  python scripts/manage_hf_collection.py --verbose

Environment Variables:
  HF_TOKEN: Required. Your Hugging Face API token.
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the collection'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 60)
    
    # Step 1: Setup API
    api = setup_api()
    
    # Step 2: Get current collection spaces
    current_spaces = get_collection_spaces(api)
    
    if args.verbose:
        logger.debug(f"Current spaces in collection: {sorted(current_spaces)}")
    
    # Step 3: Discover all openenv spaces
    discovered_spaces = discover_openenv_spaces(api)
    
    if args.verbose:
        logger.debug(f"Discovered spaces: {sorted(discovered_spaces)}")
    
    # Step 4: Find new spaces not yet in collection
    new_spaces = [s for s in discovered_spaces if s not in current_spaces]
    
    logger.info("=" * 60)
    logger.info(f"Summary:")
    logger.info(f"  Total spaces in collection: {len(current_spaces)}")
    logger.info(f"  Total spaces discovered: {len(discovered_spaces)}")
    logger.info(f"  New spaces to add: {len(new_spaces)}")
    logger.info("=" * 60)
    
    if new_spaces:
        logger.info(f"New spaces found:")
        for space in new_spaces:
            logger.info(f"  - {space}")
    
    # Step 5: Add new spaces to collection
    added_count = add_spaces_to_collection(api, new_spaces, dry_run=args.dry_run)
    
    # Final summary
    logger.info("=" * 60)
    if args.dry_run:
        logger.info(f"[DRY RUN] Would add {added_count} new spaces to collection")
    else:
        logger.info(f"✓ Successfully added {added_count} new spaces to collection")
    logger.info("=" * 60)
    
    logger.info(f"Collection URL: https://huggingface.co/collections/{COLLECTION_SLUG}")


if __name__ == "__main__":
    main()

