# Doom Environment Tests

Comprehensive test suite for the Doom environment implementation.

## Test Structure

```
tests/
├── __init__.py                  # Test package initialization
├── conftest.py                  # Shared fixtures and pytest configuration
├── test_models.py               # Unit tests for DoomAction and DoomObservation (15 tests)
├── test_doom_environment.py     # Unit tests for DoomEnvironment (18 tests)
├── test_doom_client.py          # Unit tests for DoomEnv HTTP client (20 tests)
└── test_doom_integration.py     # Integration tests (12 tests)
```

## Running Tests

### Quick Start

```bash
# Run all fast unit tests (no server required)
pytest tests/test_models.py tests/test_doom_client.py -v

# Run environment tests (requires ViZDoom)
pytest tests/test_doom_environment.py -v

# Run integration tests (requires server)
pytest tests/test_doom_integration.py -v

# Run all tests
pytest tests/ -v
```

### Test Categories

**Unit Tests (Fast)**
- `test_models.py` - Data model validation (15 tests)
- `test_doom_client.py` - HTTP client tests (20 tests)

**Environment Tests (Requires ViZDoom)**
- `test_doom_environment.py` - ViZDoom wrapper tests (18 tests)

**Integration Tests (Requires Server)**
- `test_doom_integration.py` - End-to-end tests (12 tests)

### Using Test Markers

```bash
# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow" -v

# Run only tests that require server
pytest tests/ -m "requires_server" -v

# Run only tests that don't require ViZDoom
pytest tests/ -m "not requires_vizdoom" -v

# Skip Docker tests
pytest tests/ -m "not requires_docker" -v
```

### Test Coverage

```bash
# Run with coverage report (requires pytest-cov)
pytest tests/ --cov=. --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html
```

### Continuous Integration

For CI/CD pipelines:

```bash
# Run only unit tests (no server/ViZDoom required)
pytest tests/test_models.py tests/test_doom_client.py -v

# Run with minimal output for CI logs
pytest tests/ -v --tb=short --disable-warnings
```

## Test Requirements

### Minimal (Unit Tests)
```bash
pip install pytest numpy
```

### Full (All Tests)
```bash
pip install -e .  # Installs doom_env with all dependencies
pip install pytest pytest-cov
```

## Test Fixtures

### Available Fixtures

- `sample_observation` - Pre-configured DoomObservation
- `sample_action` - Pre-configured DoomAction
- `sample_screen_buffer_160x120` - Screen buffer for 160x120 resolution
- `sample_screen_buffer_640x480` - Screen buffer for 640x480 resolution
- `doom_server` - Running Doom server (for integration tests)

### Usage Example

```python
def test_my_feature(sample_observation):
    # Use the fixture
    assert sample_observation.screen_shape == [10, 10, 3]
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from ..models import DoomAction

class TestMyFeature:
    def test_basic_functionality(self):
        """Test description."""
        action = DoomAction(action_id=0)
        assert action.action_id == 0

    @pytest.mark.slow
    def test_slow_operation(self):
        """Mark slow tests."""
        # Long-running test
        pass

    @pytest.mark.requires_vizdoom
    def test_vizdoom_feature(self):
        """Mark tests requiring ViZDoom."""
        from ..server.doom_env_environment import DoomEnvironment
        env = DoomEnvironment()
        # Test code
```

## Troubleshooting

### ImportError: ViZDoom not found
```bash
# Install ViZDoom system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libboost-all-dev libsdl2-dev

# Install ViZDoom Python package
pip install vizdoom
```

### Server connection refused
```bash
# Start the server in another terminal
python -m doom_env.server.app

# Or run integration tests with server fixture (starts automatically)
pytest tests/test_doom_integration.py -v
```

### Tests fail in headless environment
```bash
# Set environment variables for headless mode
DOOM_WINDOW_VISIBLE=false pytest tests/ -v

# Skip display-dependent tests
pytest tests/ -m "not requires_display" -v
```

## Test Metrics

**Target Coverage**: >80%

**Performance Goals**:
- Unit tests: < 10 seconds total
- Environment tests: < 30 seconds total
- Integration tests: < 2 minutes total

## Test Details

### Model Tests (test_models.py)

Tests data models without external dependencies:
- DoomAction creation, validation, serialization
- DoomObservation structure, fields, edge cases
- Numpy type conversion
- Screen buffer/shape validation
- Large buffers (800x600 RGB)

### Environment Tests (test_doom_environment.py)

Tests ViZDoom wrapper (requires ViZDoom installed):
- Environment initialization with various configs
- Reset/step functionality
- Screen buffer formats (RGB24, GRAY8)
- Multiple resolutions (160x120, 320x240, 640x480)
- Game variables tracking
- Episode termination
- Multiple scenarios support

### Client Tests (test_doom_client.py)

Tests HTTP client (uses mocks, no server needed):
- Client initialization
- Step payload serialization
- Numpy int64 conversion
- Response parsing
- Rendering modes
- Resource cleanup

### Integration Tests (test_doom_integration.py)

Tests full system (requires running server):
- Complete episode playthrough
- Client-server communication
- Reset after episode completion
- Reward accumulation
- Health tracking
- Multiple resets
- Performance benchmarks

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Add appropriate markers (`@pytest.mark.slow`, etc.)
4. Update this README if adding new test categories
5. Maintain >80% code coverage

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
- name: Run unit tests
  run: |
    pip install pytest numpy
    pytest tests/test_models.py tests/test_doom_client.py -v

- name: Run with coverage
  run: |
    pip install pytest-cov
    pytest tests/ -m "not slow" --cov=. --cov-report=xml
```
