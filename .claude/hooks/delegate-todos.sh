#!/bin/bash
# PostToolUse hook for TodoWrite: Remind about TDD workflow in worktrees

# Check if in worktree
TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null)
if [[ "$TOPLEVEL" != *".worktrees"* ]]; then
    exit 0  # Not in worktree, no reminder needed
fi

# Soft reminder about the workflow
cat << 'EOF'

TDD Workflow Reminder:
  For each todo that requires implementation:
  1. /write-tests  ->  create failing tests first
  2. /implement    ->  make tests pass
  3. Mark todo complete

EOF

exit 0
