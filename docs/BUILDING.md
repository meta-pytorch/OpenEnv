# Building the Docs Locally

## Prerequisites

- Python 3.11+

## Setup

Install OpenEnv with the docs dependencies:

```bash
pip install -e ".[docs]"
```

## Build

From the `docs/` directory:

```bash
cd docs
make html
```

The output will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Build Variants

| Command | Description |
|---------|-------------|
| `make html` | Full build with Sphinx Gallery execution |
| `make html-noplot` | Skip gallery execution (faster) |
| `make html-stable` | Build as a versioned release |
| `make clean html` | Clean rebuild from scratch |
