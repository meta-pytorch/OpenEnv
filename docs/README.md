# OpenEnv Docs Workflow

Use this guide to preview and build the Sphinx documentation that lives under `docs/`.

**Live Site:** https://meta-pytorch.org/OpenEnv

**GitHub Repo:** https://github.com/meta-pytorch/OpenEnv

## 1. Install dependencies

```bash
pip install -r docs/requirements.txt
```

Or with uv:

```bash
uv pip install -r docs/requirements.txt
```

## 2. Build the documentation

```bash
cd docs
make html
```

The built site will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to preview.

## 3. Live preview (optional)

For live reload during development, use sphinx-autobuild:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html
```

The site is served at `http://127.0.0.1:8000/` with automatic reloads.

## 4. GitHub Pages Deployment

Documentation is automatically deployed to GitHub Pages on push to `main` via `.github/workflows/docs.yml`.

## Directory Structure

```
docs/
├── source/              # Sphinx source files (canonical location)
│   ├── conf.py          # Sphinx configuration
│   ├── index.md         # Main landing page
│   ├── quickstart.md    # References ../quickstart.md via include
│   ├── cli.md           # CLI API documentation
│   ├── core.md          # Core API documentation
│   ├── environments.md  # Environments catalog
│   ├── environment-builder.md  # Build guide
│   ├── tutorials/       # Symlinks to ../tutorials/
│   └── environments/    # Symlinks to ../environments/
├── quickstart.md        # Canonical quickstart content
├── tutorials/           # Canonical tutorial content
├── environments/        # Canonical environment docs
├── build/               # Built HTML output
├── Makefile             # Build script
└── requirements.txt     # Python dependencies
```
