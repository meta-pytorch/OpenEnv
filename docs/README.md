# OpenEnv docs workflow

Use this guide to preview and build the MkDocs site that lives under `docs/`.

## 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install mkdocs-material mkdocs-include-markdown-plugin "mkdocstrings[python]" pymdown-extensions
```

The packages mirror what the GitHub Pages workflow installs, so local builds match CI.

## 2. Run the live preview server

```bash
mkdocs serve --config-file docs/mkdocs.yml
```

The site is served at `http://127.0.0.1:8000/` with automatic reloads whenever files in `docs/` or `docs/mkdocs.yml` change.

## 3. Produce the production build

```bash
mkdocs build --config-file docs/mkdocs.yml --clean --site-dir site
```

This regenerates the static HTML into `site/`, matching `.github/workflows/docs.yml`. Inspect the output locally.

