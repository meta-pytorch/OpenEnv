# OpenEnv CLI

Command-line tool for managing and deploying OpenEnv environments to Hugging Face Spaces.

> **Note**: For basic usage and examples, see the main [README.md](../../README.md#deploying-environments-to-hugging-face-spaces) in the project root. This document focuses on CLI development, testing, and architecture.

## Overview

The OpenEnv CLI provides a self-service workflow for publishing environments to Hugging Face Spaces, enabling community members to share environments without requiring GitHub PRs. The CLI handles authentication, space provisioning, building, and deployment automatically.

## Testing

### Running Tests

The CLI uses pytest for testing. Run all tests:

```bash
# From project root
pytest tests/test_cli/
```

Run specific test files:

```bash
# Test authentication
pytest tests/test_cli/test_auth.py

# Test space management
pytest tests/test_cli/test_space.py

# Test builder
pytest tests/test_cli/test_builder.py

# Test uploader
pytest tests/test_cli/test_uploader.py

# Test push command (end-to-end)
pytest tests/test_cli/test_push_command.py
```

### Test Structure

Tests follow pytest conventions and use mocking to avoid requiring actual Hugging Face API access:

- **Unit Tests**: Each module (`auth`, `space`, `builder`, `uploader`) has dedicated tests
- **Integration Tests**: `test_push_command.py` tests the full workflow with mocks
- **Fixtures**: Reusable test fixtures for mock APIs and temporary directories

### Writing Tests

Follow these patterns when adding new tests:

```python
from unittest.mock import Mock, patch
import pytest

@patch("openenv_cli.core.module.external_api_call")
def test_feature(mock_api):
    """Test description."""
    mock_api.return_value = {"result": "success"}
    
    result = function_under_test()
    
    assert result == expected_value
    mock_api.assert_called_once()
```

### Test Coverage

Run tests with coverage:

```bash
pytest --cov=openenv_cli --cov-report=html tests/test_cli/
```

## Architecture

The CLI is organized into modular components:

```
src/openenv_cli/
├── __main__.py              # CLI entry point
├── commands/
│   └── push.py              # Push command implementation
├── core/
│   ├── auth.py              # Hugging Face authentication
│   ├── space.py             # Space management (create/check)
│   ├── builder.py           # Build staging directory and files
│   └── uploader.py          # Upload to Hugging Face Spaces
└── utils/
    └── env_loader.py         # Environment validation and metadata
```

### Module Responsibilities

- **`auth.py`**: Handles Hugging Face authentication via interactive login using `huggingface_hub`
- **`space.py`**: Manages Docker Spaces (creates with `exist_ok=True` to handle existing spaces)
- **`builder.py`**: Prepares deployment package (copy files, generate Dockerfile/README)
- **`uploader.py`**: Uploads files to Hugging Face using `upload_folder`
- **`env_loader.py`**: Validates environment structure and loads metadata

## Implementation Details

### Using huggingface_hub Modules

The CLI uses `huggingface_hub` Python modules directly (not the CLI), as specified:

- `huggingface_hub.HfApi` for API calls
- `huggingface_hub.login()` for interactive authentication
- `huggingface_hub.upload_folder()` for file uploads
- `huggingface_hub.utils.get_token` for token management (reads stored credentials from previous login)

This ensures consistency with Hugging Face tooling and allows for better error handling and programmatic control. Authentication is handled exclusively through interactive login via the browser, with credentials stored by `huggingface_hub` for future use.

### Web Interface

All pushed environments automatically include the web interface:

- The Dockerfile is modified to set `ENV ENABLE_WEB_INTERFACE=true`
- The web interface is available at `/web` on deployed spaces
- Includes HumanAgent interface, state observer, and WebSocket updates

### Staging Directory

The builder creates a staging directory structure:

```
hf-staging/<env_name>/
├── Dockerfile           # Generated/modified Dockerfile
├── README.md            # With Hugging Face front matter
└── src/
    ├── core/            # Copied from src/core/
    └── envs/
        └── <env_name>/  # Copied from src/envs/<env_name>/
```

The staging directory is cleaned up after successful upload.

## Future Direction

The CLI architecture is designed to support additional commands for a complete environment management workflow:

### Planned Commands

#### `openenv init`

Initialize a new environment with the standard OpenEnv structure:

```bash
openenv init my_new_env --template coding
```

**Features:**
- Create environment directory structure
- Generate boilerplate files (models.py, client.py, server/)
- Initialize Dockerfile and app.py
- Create README template

#### `openenv upgrade`

Upgrade an existing environment to a new OpenEnv version:

```bash
openenv upgrade my_env --to-version 0.2.0
```

**Features:**
- Check for breaking changes
- Update dependencies
- Migrate code to new API versions
- Backup existing environment

#### `openenv test`

Run local tests before pushing:

```bash
openenv test my_env
# or
openenv test my_env --docker
```

**Features:**
- Run environment unit tests
- Test Docker build process
- Validate environment structure
- Check for common issues

#### `openenv validate`

Pre-submission validation:

```bash
openenv validate my_env
```

**Features:**
- Validate environment structure
- Check Dockerfile syntax
- Verify README formatting
- Test web interface generation
- Validate models and client code

### Extension Points

The CLI architecture supports extension through:

1. **Command Registration**: Add new commands in `commands/` and register in `__main__.py`
2. **Core Modules**: Add new core functionality (e.g., `core/validator.py`)
3. **Shared Utilities**: Extend `utils/` for reusable functions

### Example: Adding a New Command

```python
# src/openenv_cli/commands/validate.py
def validate_environment(env_name: str) -> None:
    """Validate environment structure."""
    # Implementation
    pass

# src/openenv_cli/__main__.py
# Add to subparsers:
validate_parser = subparsers.add_parser("validate", help="Validate environment")
validate_parser.add_argument("env_name")
# ... handle command
```

## Contributing

When adding new features:

1. **Write Tests First**: Follow TDD approach, write tests before implementation
2. **Use Mocks**: Mock external APIs (Hugging Face) to keep tests fast and isolated
3. **Follow Patterns**: Match existing code style and patterns
4. **Update Documentation**: Update this README and add docstrings

## References

- [Hugging Face Hub Python API](https://huggingface.co/docs/huggingface_hub/index)
- [OpenEnv Environment Structure](../envs/README.md)
- [OpenEnv RFCs](../../rfcs/)
