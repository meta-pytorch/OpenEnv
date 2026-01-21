---
name: write-tests
description: Write failing tests from requirements. Invoke for each todo before /implement.
context: fork
agent: tester
---

# /write-tests

Write failing tests that encode acceptance criteria.

## Usage

```
/write-tests
/write-tests Add logout button to header
```

## When to Use

- After creating a todo that requires implementation
- Before running `/implement`
- When you have clear acceptance criteria

## When NOT to Use

- Implementation already exists (tests would pass immediately)
- You're exploring or prototyping (not TDD mode)
- Just adding to existing test coverage

## What It Does

1. Analyzes the current todo/requirement
2. Reads existing tests to understand patterns
3. Writes test files that verify acceptance criteria
4. **Verifies tests FAIL** (proves they test something real)
5. Returns test file paths for `/implement`

## Output

The tester agent will produce:

```markdown
## Tests Written

### Files Created/Modified
- `tests/test_client.py`

### Tests Added
| Test | Verifies |
|------|----------|
| `test_client_reset_returns_observation` | Reset returns valid observation |
| `test_client_step_advances_state` | Step mutates state correctly |
| `test_client_handles_invalid_action` | Error handling for bad input |

### Verification
All tests FAIL as expected (no implementation yet).

### Next Step
Run `/implement` to make these tests pass.
```

## Rules

1. **Read existing tests first** to understand patterns and conventions
2. **Test behavior, not implementation** - write from user's perspective
3. **Integration tests first**, then unit tests if needed
4. **Each test verifies ONE thing** clearly
5. **Run tests to verify they fail** before returning

## Anti-patterns (NEVER do these)

- Writing tests that pass without implementation
- Testing implementation details instead of behavior
- Writing overly complex test setups
- Adding implementation code (that's `/implement`'s job)
- Writing tests that duplicate existing coverage

## Completion Criteria

Before returning, verify:
1. Tests compile/run successfully (pytest can collect them)
2. Tests FAIL (no implementation yet)
3. Test names clearly describe what they verify
4. Tests follow existing project patterns (see `tests/` for examples)
