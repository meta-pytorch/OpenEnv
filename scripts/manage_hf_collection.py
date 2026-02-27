#!/usr/bin/env python3
"""
Hugging Face Collection Manager for OpenEnv

This script can:
1) add an explicit list of deployed OpenEnv spaces to a collection, or
2) discover OpenEnv spaces by tag and add missing ones.

Collections can be version-specific (for example: "OpenEnv Environment Hub v2.1.0").

Usage:
    python scripts/manage_hf_collection.py [--dry-run] [--verbose]
    python scripts/manage_hf_collection.py --version v2.1.0 --space-id openenv/echo_env

Environment Variables:
    HF_TOKEN: Optional. If omitted, local Hugging Face auth is used.
"""

import argparse
import logging
import os
import sys
from typing import Iterable, List, Optional, Set

from huggingface_hub import HfApi, list_spaces
from huggingface_hub.utils import HfHubHTTPError


# Constants
DEFAULT_COLLECTION_SLUG = "openenv/environment-hub-68f16377abea1ea114fa0743"
DEFAULT_COLLECTION_NAMESPACE = "openenv"
DEFAULT_COLLECTION_TITLE_PREFIX = "OpenEnv Environment Hub"
DEFAULT_TAG_FILTER = "openenv"
DEFAULT_VERSION = "v2.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_api() -> HfApi:
    """
    Initialize and authenticate the Hugging Face API client.

    Returns:
        HfApi: Authenticated API client

    Raises:
        SystemExit: If authentication fails
    """
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Authenticating with Hugging Face...")
    api = HfApi(token=hf_token) if hf_token else HfApi()

    try:
        whoami = api.whoami()
        logger.info(f"✓ Authenticated as: {whoami['name']}")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        logger.error(
            "Set HF_TOKEN or run 'hf auth login' before running this script."
        )
        sys.exit(1)

    return api


def get_collection_spaces(api: HfApi, collection_slug: str) -> Set[str]:
    """
    Retrieve the list of spaces currently in a collection.

    Args:
        api: Authenticated HfApi client
        collection_slug: Hugging Face collection slug

    Returns:
        Set of space IDs (in format "owner/space-name")
    """
    logger.info(f"Fetching current collection: {collection_slug}")

    try:
        collection = api.get_collection(collection_slug)

        # Extract space IDs from collection items
        space_ids = set()
        for item in collection.items:
            if item.item_type == "space":
                space_ids.add(item.item_id)

        logger.info(f"✓ Found {len(space_ids)} spaces in collection")
        return space_ids

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            logger.error(f"Collection not found: {collection_slug}")
            logger.error("Please ensure the collection exists and you have access to it")
        else:
            logger.error(f"Error fetching collection: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error fetching collection: {e}")
        sys.exit(1)


def normalize_version(version: str) -> str:
    """Normalize version strings for collection titles."""
    return version.strip()


def build_collection_title(prefix: str, version: str) -> str:
    """Build a predictable versioned collection title."""
    return f"{prefix} {normalize_version(version)}"


def find_collection_by_title(
    api: HfApi, namespace: str, title: str
) -> Optional[str]:
    """Find a collection slug by exact title within a namespace."""
    try:
        collections = api.list_collections(owner=namespace)
    except Exception as e:
        logger.error(f"Failed to list collections for '{namespace}': {e}")
        return None

    for collection in collections:
        if getattr(collection, "title", None) == title:
            return collection.slug
    return None


def resolve_collection_slug(
    api: HfApi,
    version: str,
    namespace: str,
    title_prefix: str,
    explicit_slug: Optional[str],
    dry_run: bool,
) -> str:
    """
    Resolve target collection slug.

    Priority:
    1) explicit --collection-slug
    2) existing or newly created versioned collection
    """
    if explicit_slug:
        return explicit_slug

    title = build_collection_title(title_prefix, version)

    existing_slug = find_collection_by_title(api, namespace, title)
    if existing_slug:
        logger.info(f"Using existing versioned collection: {existing_slug}")
        return existing_slug

    if dry_run:
        synthetic = f"{namespace}/dry-run-{version.replace('.', '-')}"
        logger.info(
            f"[DRY RUN] Collection '{title}' not found; would create it under '{namespace}'"
        )
        return synthetic

    try:
        collection = api.create_collection(
            title=title,
            namespace=namespace,
            description=f"OpenEnv spaces for release {version}",
            exists_ok=True,
        )
        logger.info(f"✓ Created/using versioned collection: {collection.slug}")
        return collection.slug
    except Exception as e:
        logger.error(f"Failed to create versioned collection '{title}': {e}")
        logger.error("Pass --collection-slug to target an existing collection explicitly.")
        sys.exit(1)


def discover_openenv_spaces(api: HfApi, tag_filter: str) -> List[str]:
    """
    Search for all Docker Spaces tagged with 'openenv'.

    Args:
        api: Authenticated HfApi client
        tag_filter: tag that identifies OpenEnv spaces

    Returns:
        List of space IDs (in format "owner/space-name")
    """
    logger.info(f"Searching for Docker Spaces with tag '{tag_filter}'...")

    try:
        spaces = list(
            list_spaces(
                search=tag_filter,
                full=False,
                sort="trending_score",
                direction=-1,
            )
        )

        docker_spaces_with_tag: List[str] = []
        for space in spaces:
            try:
                space_info = api.space_info(space.id)
                runtime = getattr(space_info, "runtime", None)
                runtime_stage = getattr(runtime, "stage", None)
                tags = getattr(space_info, "tags", []) or []
                if (
                    getattr(space_info, "sdk", None) == "docker"
                    and tag_filter in tags
                    and runtime_stage != "RUNTIME_ERROR"
                ):
                    docker_spaces_with_tag.append(space.id)
            except Exception as e:
                logger.warning(f"Could not fetch info for space {space.id}: {e}")
                continue

        logger.info(
            f"✓ Discovered {len(docker_spaces_with_tag)} Docker spaces with tag '{tag_filter}'"
        )
        return docker_spaces_with_tag

    except Exception as e:
        logger.error(f"Error discovering spaces: {e}")
        sys.exit(1)


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Deduplicate while preserving original order."""
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def add_spaces_to_collection(
    api: HfApi,
    collection_slug: str,
    space_ids: List[str],
    version: str,
    dry_run: bool = False
) -> int:
    """
    Add new spaces to a collection.

    Args:
        api: Authenticated HfApi client
        collection_slug: Collection slug to update
        space_ids: List of space IDs to add
        version: OpenEnv version label for collection notes
        dry_run: If True, only simulate the addition without making changes

    Returns:
        Number of spaces added (or would be added in dry-run mode)
    """
    if not space_ids:
        logger.info("No new spaces to add")
        return 0
    
    added_count = 0
    failed_count = 0
    note = f"OpenEnv release {version}"

    for space_id in space_ids:
        if dry_run:
            logger.info(f"[DRY RUN] Would add space to {collection_slug}: {space_id}")
            added_count += 1
        else:
            try:
                logger.info(f"Adding space to collection: {space_id}")
                api.add_collection_item(
                    collection_slug=collection_slug,
                    item_id=space_id,
                    item_type="space",
                    note=note,
                    exists_ok=True,
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
        description="Manage versioned Hugging Face collections for OpenEnv spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add explicit deployed spaces to collection for a release
  python scripts/manage_hf_collection.py --version v2.1.0 \\
      --space-id openenv/echo_env --space-id openenv/coding_env

  # Discover spaces by tag and add them to a versioned collection
  python scripts/manage_hf_collection.py --version v2.1.0 --verbose

Environment Variables:
  HF_TOKEN: Optional. If unset, local HF auth credentials are used.
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the collection",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Version label for release/collection (default: {DEFAULT_VERSION})",
    )

    parser.add_argument(
        "--space-id",
        action="append",
        default=[],
        help="Space to add (repeatable, e.g. openenv/echo_env). If omitted, discovery mode is used.",
    )

    parser.add_argument(
        "--collection-slug",
        default=None,
        help=(
            "Explicit collection slug. If omitted, the script uses or creates a "
            "versioned collection."
        ),
    )

    parser.add_argument(
        "--collection-namespace",
        default=DEFAULT_COLLECTION_NAMESPACE,
        help=(
            "Namespace to own versioned collections when --collection-slug is not "
            f"provided (default: {DEFAULT_COLLECTION_NAMESPACE})."
        ),
    )

    parser.add_argument(
        "--collection-title-prefix",
        default=DEFAULT_COLLECTION_TITLE_PREFIX,
        help=(
            "Prefix used for versioned collection titles when "
            "--collection-slug is not provided."
        ),
    )

    parser.add_argument(
        "--tag-filter",
        default=DEFAULT_TAG_FILTER,
        help=(
            "Tag used in discovery mode (when --space-id is not supplied). "
            f"Default: {DEFAULT_TAG_FILTER}"
        ),
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

    # Step 2: Resolve collection
    collection_slug = resolve_collection_slug(
        api=api,
        version=args.version,
        namespace=args.collection_namespace,
        title_prefix=args.collection_title_prefix,
        explicit_slug=args.collection_slug,
        dry_run=args.dry_run,
    )

    # Step 3: Get current collection spaces
    if args.dry_run and collection_slug.endswith(f"/dry-run-{args.version.replace('.', '-')}"):
        current_spaces = set()
        logger.info("[DRY RUN] Synthetic collection slug in use; skipping collection fetch.")
    else:
        current_spaces = get_collection_spaces(api, collection_slug)

    if args.verbose:
        logger.debug(f"Current spaces in collection: {sorted(current_spaces)}")

    # Step 4: Resolve target spaces
    if args.space_id:
        discovered_spaces = dedupe_preserve_order(args.space_id)
        logger.info(f"Using explicit space IDs ({len(discovered_spaces)}): {discovered_spaces}")
    else:
        discovered_spaces = discover_openenv_spaces(api, args.tag_filter)

    if args.verbose:
        logger.debug(f"Discovered spaces: {sorted(discovered_spaces)}")

    # Step 5: Find new spaces not yet in collection
    new_spaces = [s for s in discovered_spaces if s not in current_spaces]

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Version: {args.version}")
    logger.info(f"  Collection: {collection_slug}")
    logger.info(f"  Total spaces in collection: {len(current_spaces)}")
    logger.info(f"  Total spaces resolved: {len(discovered_spaces)}")
    logger.info(f"  New spaces to add: {len(new_spaces)}")
    logger.info("=" * 60)

    if new_spaces:
        logger.info("New spaces found:")
        for space in new_spaces:
            logger.info(f"  - {space}")

    # Step 6: Add new spaces to collection
    added_count = add_spaces_to_collection(
        api=api,
        collection_slug=collection_slug,
        space_ids=new_spaces,
        version=args.version,
        dry_run=args.dry_run,
    )

    # Final summary
    logger.info("=" * 60)
    if args.dry_run:
        logger.info(f"[DRY RUN] Would add {added_count} new spaces to collection")
    else:
        logger.info(f"✓ Successfully added {added_count} new spaces to collection")
    logger.info("=" * 60)

    logger.info(f"Collection URL: https://huggingface.co/collections/{collection_slug}")


if __name__ == "__main__":
    main()
