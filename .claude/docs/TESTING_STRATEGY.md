# OpenEnv Testing Strategy

This document outlines OpenEnv's testing philosophy, hierarchy, and conventions.

## Testing Hierarchy

Tests are organized by scope and signal:

### 1. Unit Tests (Fastest, Most Isolated)

Test individual functions and classes in isolation.

**Good candidates:**
- Pure functions (e.g., reward calculations)
- Pydantic model validation
- State mutations
- Utility functions

**Location:** `tests/` mirroring `src/` structure

**Example:**
```python
def test_action_model_validates_required_fields():
    with pytest.raises(ValidationError):
        Action()  # Missing required fields
```

### 2. Integration Tests (Medium Scope)

Test component interactions, especially client-server communication.

**Good candidates:**
- Client-server WebSocket protocol
- Environment lifecycle (reset → step → step → ...)
- Type serialization across wire boundary

**Location:** `tests/` with `_integration` suffix or in dedicated directories

**Example:**
```python
async def test_client_connects_and_resets():
    async with start_server() as server:
        client = EchoEnvClient(server.url)
        obs = await client.reset()
        assert isinstance(obs, EchoObservation)
```

### 3. Environment Validation Tests

Test that environments follow OpenEnv conventions and invariants.

**Good candidates:**
- File structure validation
- Type consistency (generics match)
- Invariant checking (no client→server imports)

**Location:** `tests/envs/`

**Uses:** `env-validator` agent patterns

### 4. E2E Tests (Slowest, Highest Signal)

Test complete workflows from user perspective.

**Good candidates:**
- Full training loop simulation
- Container lifecycle
- MCP tool interactions

**Location:** `tests/e2e/` (if needed)

## Test Location Conventions

```
tests/
├── conftest.py              # Shared fixtures
├── core/                    # Core library tests
│   ├── test_environment.py
│   ├── test_client.py
│   └── test_server.py
├── envs/                    # Environment-specific tests
│   ├── test_echo_environment.py
│   └── test_<env>_environment.py
└── e2e/                     # End-to-end tests (optional)
```

## Running Tests

### Full Suite
```bash
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short
```

### Single File
```bash
PYTHONPATH=src:envs uv run pytest tests/path/test_file.py -v
```

### Single Test
```bash
PYTHONPATH=src:envs uv run pytest tests/path/test_file.py::test_name -v
```

### Exclude Special Environments
Some environments require special setup (browser, websearch). The hook script excludes these:
```bash
bash .claude/hooks/test.sh
```

## Edge Cases to Consider

### Python-Specific
- `None` where not expected
- Type mismatches at runtime (despite type hints)
- Pydantic `ValidationError` on invalid data
- Async/await edge cases (timeouts, cancellation)

### State Management
- Empty state / default values
- Maximum capacity / overflow
- State after error recovery
- Concurrent access patterns

### Protocol / WebSocket
- Connection drops mid-step
- Out-of-order messages
- Malformed JSON payloads
- Timeout handling

### Pydantic Models
- Extra fields in input (strict mode)
- Missing required fields
- Type coercion behavior
- Nested model validation

## Test Patterns

### Fixtures for Common Setup

```python
@pytest.fixture
def echo_env():
    """Create a fresh EchoEnvironment for each test."""
    return EchoEnvironment()

def test_reset_returns_observation(echo_env):
    obs, _ = echo_env.reset()
    assert isinstance(obs, EchoObservation)
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_client():
    async with create_client() as client:
        result = await client.step(action)
        assert result.done is False
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("", ""),
    ("123", "123"),
])
def test_transform(input, expected):
    assert transform(input) == expected
```

## What Makes a Good Test

### High-Signal (Write These)

- Catches bugs that could happen in production
- Tests behavior from user perspective
- Covers non-obvious edge cases
- Validates complex state machines

### Low-Signal (Avoid These)

- Tests that verify Python built-ins work
- Duplicates of existing tests with trivial variation
- Tests that mock so much they don't test real behavior
- Tests for code paths already covered by integration tests

## TDD Workflow

The testing strategy integrates with the TDD workflow:

1. **Red**: `/write-tests` creates failing tests
2. **Green**: `/implement` makes tests pass
3. **Refactor**: `/simplify` cleans up code
4. **Validate**: `/pre-submit-pr` runs full suite

## Coverage Gaps (Known)

Document known gaps here as they're identified:

- [ ] WebSocket reconnection handling
- [ ] Container lifecycle edge cases
- [ ] MCP tool error responses (when MCP is added)

## Verification

After writing tests, verify with:

```bash
# Run specific tests
PYTHONPATH=src:envs uv run pytest tests/path/test_file.py -v

# Check coverage (if coverage is set up)
PYTHONPATH=src:envs uv run pytest tests/ --cov=src/openenv

# Run lint to ensure test code is clean
uv run ruff check tests/
```
