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

The output will be in `docs/_build/html/`.

## Preview

From the repo root, start a local server:

```bash
cd docs/_build/html
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

### Build Variants

| Command | Description |
|---------|-------------|
| `make html` | Full build with Sphinx Gallery execution |
| `make html-noplot` | Skip gallery execution (faster) |
| `make html-stable` | Build as a versioned release |
| `make clean html` | Clean rebuild from scratch |
