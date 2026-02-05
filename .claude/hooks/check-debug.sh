#!/bin/bash
# Check for debug code that shouldn't be committed
# Exit code 0 always (informational), but outputs findings

# Check for required tools
if ! command -v rg &> /dev/null; then
    echo "Warning: 'rg' (ripgrep) is not installed, falling back to grep"
    USE_GREP=1
fi

echo "=== Checking for debug code ==="

found_issues=0

# Check for print statements (allow if marked with # ok-to-print)
echo ""
echo "--- Print statements in src/ ---"
if [ "$USE_GREP" = "1" ]; then
    prints=$(grep -rn "print(" src/ --include="*.py" 2>/dev/null | grep -v "# ok-to-print" || true)
else
    prints=$(rg -n "print\(" src/ --glob "*.py" 2>/dev/null | grep -v "# ok-to-print" || true)
fi

if [ -n "$prints" ]; then
    echo "$prints"
    found_issues=1
else
    echo "None found"
fi

# Check for TODO/FIXME/XXX/HACK comments
echo ""
echo "--- TODO/FIXME comments in src/ ---"
if [ "$USE_GREP" = "1" ]; then
    todos=$(grep -rn -E "TODO|FIXME|XXX|HACK" src/ --include="*.py" 2>/dev/null || true)
else
    todos=$(rg -n "TODO|FIXME|XXX|HACK" src/ --glob "*.py" 2>/dev/null || true)
fi

if [ -n "$todos" ]; then
    echo "$todos"
    found_issues=1
else
    echo "None found"
fi

# Check for debugger statements
echo ""
echo "--- Debugger statements in src/ ---"
if [ "$USE_GREP" = "1" ]; then
    debuggers=$(grep -rn -E "breakpoint\(\)|pdb\.|ipdb\." src/ --include="*.py" 2>/dev/null || true)
else
    debuggers=$(rg -n "breakpoint\(\)|pdb\.|ipdb\." src/ --glob "*.py" 2>/dev/null || true)
fi

if [ -n "$debuggers" ]; then
    echo "$debuggers"
    found_issues=1
else
    echo "None found"
fi

echo ""
if [ $found_issues -eq 1 ]; then
    echo "=== Debug code found (review before committing) ==="
else
    echo "=== No debug code found ==="
fi

# Always exit 0 - this is informational
exit 0
