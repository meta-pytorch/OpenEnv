#!/bin/bash
# Create a git worktree for a new feature branch
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <branch-name>"
    echo ""
    echo "Example: $0 add-auth"
    echo "Creates: .worktrees/add-auth with branch feature/add-auth"
    exit 1
fi

BRANCH_NAME="$1"
FEATURE_BRANCH="feature/$BRANCH_NAME"

# Get repo root
REPO_ROOT=$(git rev-parse --show-toplevel)

# Worktree path is inside .worktrees/ subdirectory
WORKTREE_PATH="$REPO_ROOT/.worktrees/$BRANCH_NAME"

# Ensure .worktrees directory exists
mkdir -p "$REPO_ROOT/.worktrees"

# Check if worktree already exists
if [ -d "$WORKTREE_PATH" ]; then
    echo "ERROR: Worktree already exists at $WORKTREE_PATH"
    exit 1
fi

# Check if branch already exists
if git show-ref --verify --quiet "refs/heads/$FEATURE_BRANCH"; then
    echo "Branch $FEATURE_BRANCH already exists, using existing branch"
    git worktree add "$WORKTREE_PATH" "$FEATURE_BRANCH"
else
    echo "Creating new branch $FEATURE_BRANCH"
    git worktree add -b "$FEATURE_BRANCH" "$WORKTREE_PATH"
fi

echo ""
echo "Worktree created successfully!"
echo ""
echo "Path: $WORKTREE_PATH"
echo "Branch: $FEATURE_BRANCH"
echo ""
echo "To start working:"
echo "  cd .worktrees/$BRANCH_NAME"
