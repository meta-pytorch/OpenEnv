# Installation

!!! note "Coming Soon"
    This page is under construction. For now, see the [Quick Start](quickstart.md) for basic installation instructions.

## Quick Install

```bash
pip install openenv-core
```

## Installation Options

### From PyPI (Recommended)

```bash
pip install openenv-core
```

### Using uv

```bash
uv pip install openenv-core
```

### From Source

```bash
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv
pip install -e .
```

### Installing Environments

Individual environments can be installed separately:

```bash
pip install openenv-coding-env
pip install openenv-chess-env
pip install openenv-browsergym-env
```

Or install multiple environments at once:

```bash
pip install openenv-core[coding,chess,browsergym]
```

## Requirements

- Python 3.10+
- Docker (optional, for running environments locally)

## Next Steps

- [Quick Start](quickstart.md) - Get started in 5 minutes
- [Core Concepts](concepts.md) - Understand the key abstractions
