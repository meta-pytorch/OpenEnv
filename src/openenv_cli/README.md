# OpenEnv CLI

Command-line tool for managing and deploying OpenEnv environments to HuggingFace Spaces.

## Overview

The OpenEnv CLI provides a self-service workflow for publishing environments to HuggingFace Spaces, enabling community members to share environments without requiring GitHub PRs. The CLI handles authentication, space provisioning, building, and deployment automatically.

## Installation

The CLI is installed as part of the OpenEnv package:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Push Environment

Push an environment to HuggingFace Spaces:

```bash
openenv push <env_name> [options]
```

**Arguments:**
- `env_name`: Name of the environment to push (e.g., `echo_env`, `coding_env`)

**Options:**
- `--namespace <namespace>`: HuggingFace namespace (organization or user). If not provided, uses authenticated user's username.
- `--space-name <space_name>`: Custom name for the HuggingFace Space. If not provided, uses the environment name.
- `--private`: Create a private space (default: public)
- `--base-image <image>`: Base Docker image to use (default: `ghcr.io/meta-pytorch/openenv-base:latest`)
- `--dry-run`: Prepare files but don't upload to HuggingFace

**Examples:**

```bash
# Push echo_env to your personal namespace
openenv push echo_env

# Push to a specific organization
openenv push coding_env --namespace my-org

# Push with a custom space name
openenv push echo_env --space-name my-custom-space

# Push to an organization with a custom space name
openenv push echo_env --namespace my-org --space-name my-custom-space

# Create a private space
openenv push echo_env --private

# Use a custom base image
openenv push echo_env --base-image ghcr.io/my-org/custom-base:latest

# Prepare files without uploading
openenv push echo_env --dry-run
```

### Authentication

The CLI uses HuggingFace authentication. You can authenticate in two ways:

1. **Environment Variable**: Set `HF_TOKEN` environment variable with your HuggingFace token
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **Interactive Login**: If no token is found, the CLI will prompt for interactive login:
   ```bash
   # The CLI will automatically prompt when needed
   openenv push echo_env
   ```

To get a token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "write" permissions
3. Use it as `HF_TOKEN` or log in interactively

## How It Works

The `openenv push` command performs the following steps:

1. **Validation**: Checks that the environment exists in `src/envs/<env_name>/`
2. **Authentication**: Ensures you're authenticated with HuggingFace (prompts if needed)
3. **Space Provisioning**: Determines the target Space name (uses `--space-name` if provided, otherwise `env_name`) and namespace (`--namespace` if provided, otherwise authenticated user). Checks if a Docker Space exists, creates it if needed
4. **Build Process**:
   - Creates a staging directory
   - Copies core and environment files
   - Generates/modifies Dockerfile with web interface enabled
   - Prepares README: If the environment's README already has HuggingFace front matter (starts and ends with `---`), uses it as-is. Otherwise, generates front matter with random emoji and colors from approved options
5. **Deployment**: Uploads all files to the HuggingFace Space
6. **Cleanup**: Removes staging directory after successful upload

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

Tests follow pytest conventions and use mocking to avoid requiring actual HuggingFace API access:

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
│   ├── auth.py              # HuggingFace authentication
│   ├── space.py             # Space management (create/check)
│   ├── builder.py           # Build staging directory and files
│   └── uploader.py          # Upload to HuggingFace Spaces
└── utils/
    └── env_loader.py         # Environment validation and metadata
```

### Module Responsibilities

- **`auth.py`**: Handles HuggingFace authentication using `huggingface_hub`
- **`space.py`**: Manages Docker Spaces (check existence, create)
- **`builder.py`**: Prepares deployment package (copy files, generate Dockerfile/README)
- **`uploader.py`**: Uploads files to HuggingFace using `upload_folder`
- **`env_loader.py`**: Validates environment structure and loads metadata

## Implementation Details

### Using huggingface_hub Modules

The CLI uses `huggingface_hub` Python modules directly (not the CLI), as specified:

- `huggingface_hub.HfApi` for API calls
- `huggingface_hub.login()` for authentication
- `huggingface_hub.upload_folder()` for file uploads
- `huggingface_hub.utils.get_token` for token management

This ensures consistency with HuggingFace tooling and allows for better error handling and programmatic control.

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
├── README.md            # With HuggingFace front matter
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

## Troubleshooting

### Authentication Issues

**Problem**: "Failed to retrieve token after login"

**Solution**: 
- Check that `huggingface_hub` is properly installed: `pip install --upgrade huggingface_hub`
- Verify token permissions (needs "write" access)
- Try logging in via CLI: `huggingface-cli login`

### Space Creation Fails

**Problem**: "Failed to create space"

**Solution**:
- Check that namespace/username is correct
- Verify you have permission to create spaces in that namespace
- Ensure space name doesn't already exist (check on huggingface.co)

### Upload Fails

**Problem**: "Failed to upload to space"

**Solution**:
- Check internet connection
- Verify token hasn't expired
- Try `--dry-run` first to check file preparation
- Check staging directory exists and has files

### Environment Not Found

**Problem**: "Environment 'xyz' not found"

**Solution**:
- Verify environment exists in `src/envs/<env_name>/`
- Check spelling of environment name
- Ensure environment directory has required structure (models.py, server/, etc.)

## Contributing

When adding new features:

1. **Write Tests First**: Follow TDD approach, write tests before implementation
2. **Use Mocks**: Mock external APIs (HuggingFace) to keep tests fast and isolated
3. **Follow Patterns**: Match existing code style and patterns
4. **Update Documentation**: Update this README and add docstrings

## References

- [HuggingFace Hub Python API](https://huggingface.co/docs/huggingface_hub/index)
- [OpenEnv Environment Structure](../envs/README.md)
- [OpenEnv RFCs](../../rfcs/)
