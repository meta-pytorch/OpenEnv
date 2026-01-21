#!/bin/bash
# PreToolUse hook for Edit/Write: Block direct code edits in worktree mode
#
# Design: Only block in worktrees (focused work mode).
# In main repo, allow direct edits (exploration mode).

# Check if in worktree
TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null)
if [[ "$TOPLEVEL" != *".worktrees"* ]]; then
    exit 0  # Not in worktree, allow all edits
fi

# Read JSON from stdin (hook input format)
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

# If no file path or jq failed, allow
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Only check Python implementation files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0  # Not a Python file, allow
fi

# Allow test files
if [[ "$FILE_PATH" == *test* ]] || [[ "$FILE_PATH" == */tests/* ]]; then
    exit 0  # Test file, allow (tester persona can write these)
fi

# Allow non-src files (scripts, configs, etc.)
if [[ "$FILE_PATH" != */src/* ]] && [[ "$FILE_PATH" != */envs/* ]]; then
    exit 0
fi

# Block with helpful message
cat >&2 << 'EOF'

===================================================================
  WORKTREE MODE: Direct code edit blocked
===================================================================

In worktrees, use the TDD workflow:

  1. /write-tests  ->  tester writes failing tests
  2. /implement    ->  implementer makes tests pass

To bypass this check, say "skip TDD" in your message.

===================================================================

EOF

exit 2
