---
name: implement
description: Make tests pass. Invoke after /write-tests produces failing tests.
context: fork
agent: implementer
---

# /implement

Make failing tests pass with minimal code.

## Usage

```
/implement
```

The implementer will automatically find failing tests from the most recent `/write-tests` run.

## When to Use

- After `/write-tests` has created failing tests
- When you have specific tests that need implementation
- Never before tests exist

## When NOT to Use

- No failing tests exist (run `/write-tests` first)
- You want to add features not covered by tests
- You want to refactor (use `/simplify` instead)

## What It Does

1. Finds the failing tests from `/write-tests`
2. Reads tests to understand requirements
3. Writes the **minimum code** to make tests pass
4. Runs tests after each change
5. Stops when ALL tests pass

## Output

The implementer agent will produce:

```markdown
## Implementation Complete

### Tests Passed
- `test_client_reset_returns_observation` ✓
- `test_client_step_advances_state` ✓
- `test_client_handles_invalid_action` ✓

### Changes Made
| File | Change |
|------|--------|
| `src/openenv/core/client.py` | Added `reset()` method |
| `src/openenv/core/client.py` | Added `step()` method |
| `src/openenv/core/client.py` | Added input validation |

### Verification
```
PYTHONPATH=src:envs uv run pytest tests/test_client.py -v
   All 3 tests passed
```

### Next Steps
- Mark todo as complete
- Consider `/simplify` if change was large
- Move to next pending todo
```

## Rules

1. **Read the failing tests first** to understand exactly what's needed
2. **Write the MINIMUM code** needed to pass tests
3. **Run tests after each change** to verify progress
4. **Do NOT add extra features** not covered by tests
5. **Do NOT refactor** existing code (that's `/simplify`'s job)
6. **Stop when all tests pass**

## Anti-patterns (NEVER do these)

- Adding features not covered by tests
- Refactoring existing code
- Writing additional tests (that's `/write-tests`'s job)
- Over-engineering solutions
- Adding comments or documentation beyond what's necessary
- "Improving" code that already works

## Completion Criteria

Before returning, verify:
1. ALL tests pass
2. No new test failures introduced
3. Implementation is minimal and focused

## Philosophy

The implementer is a "code machine" - it takes test specifications and produces the minimal code to satisfy them. This keeps implementations focused and prevents scope creep.

Think of it as TDD's second phase: Red → **Green** → Refactor. You are "Green" - make tests pass, nothing more.
