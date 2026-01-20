#!/bin/bash
# SessionStart hook: Show context and set mode based on location

echo ""

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    exit 0
fi

TOPLEVEL=$(git rev-parse --show-toplevel)

# Check if in worktree
if [[ "$TOPLEVEL" == *".worktrees"* ]]; then
    FEATURE=$(basename "$TOPLEVEL")
    echo "==================================================================="
    echo "  WORKTREE: $FEATURE"
    echo "==================================================================="

    # Try to find issue number from branch name
    BRANCH=$(git branch --show-current 2>/dev/null)
    if [[ "$BRANCH" =~ ([0-9]+) ]]; then
        echo "  Issue: #${BASH_REMATCH[1]}"
    fi
    echo "  Branch: $BRANCH"
    echo ""
    echo "  TDD MODE ACTIVE - Direct code edits blocked"
    echo ""
    echo "  Workflow:"
    echo "    /write-tests  ->  create failing tests"
    echo "    /implement    ->  make tests pass"
    echo "    /simplify     ->  clean up (optional)"
    echo "    /pre-submit-pr   ->  validate before commit"
    echo ""
    echo "  Say \"skip TDD\" to bypass blocking"
    echo "==================================================================="
else
    echo "==================================================================="
    echo "  MAIN REPO (Explore Mode)"
    echo "==================================================================="
    echo ""
    echo "  Direct edits allowed. For focused work:"
    echo "    /work-on-issue #42  ->  start TDD workflow"
    echo ""
    echo "  Or manually:"
    echo "    .claude/scripts/worktree-create.sh <name>"
    echo "==================================================================="
fi

echo ""
