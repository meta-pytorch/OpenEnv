#!/bin/bash
# Clean up a git worktree after PR is merged
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <worktree-path>"
    echo ""
    echo "Example: $0 .worktrees/add-auth"
    echo "Removes the worktree and optionally deletes the branch"
    exit 1
fi

WORKTREE_PATH="$1"

# Verify it's a valid worktree
if [ ! -d "$WORKTREE_PATH" ]; then
    echo "ERROR: Directory does not exist: $WORKTREE_PATH"
    exit 1
fi

if [ ! -f "$WORKTREE_PATH/.git" ]; then
    echo "ERROR: Not a git worktree: $WORKTREE_PATH"
    exit 1
fi

# Get the branch name
cd "$WORKTREE_PATH"
BRANCH=$(git branch --show-current)
cd - > /dev/null

echo "Removing worktree: $WORKTREE_PATH"
echo "Branch: $BRANCH"
echo ""

# Remove the worktree
git worktree remove "$WORKTREE_PATH" --force

echo "Worktree removed."
echo ""

# Ask about branch deletion
read -p "Delete branch '$BRANCH'? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git branch -D "$BRANCH"
    echo "Branch deleted."
else
    echo "Branch kept. Delete manually with: git branch -D $BRANCH"
fi

echo ""
echo "Cleanup complete!"
