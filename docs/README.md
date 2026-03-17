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

## How the Getting Started Section Builds

The Getting Started section is powered by [Sphinx Gallery](https://sphinx-gallery.github.io/). Source files live in `docs/source/getting_started/`:

- **`plot_*.py`** — executable Python scripts that Sphinx Gallery converts into rendered notebook pages
- **`*.md`** — static Markdown pages (environment-builder, contributing-envs)
- **`README.rst`** — gallery index template

During the build, Sphinx Gallery processes these sources and writes the output into a generated `auto_getting_started/` directory. A custom `copy_md_pages_to_gallery` hook in `conf.py` copies the static `.md` pages into that same output directory so they appear alongside the gallery notebooks in the left nav.

Because `getting_started/*.md` is in `exclude_patterns` in `conf.py`, Sphinx only generates HTML from the `auto_getting_started/` output — not from the source directory directly. This means all internal links to Getting Started pages must use the `auto_getting_started/` path (e.g. `auto_getting_started/environment-builder.md`). Linking to `getting_started/` will 404.

The copy hook runs on the `builder-inited` event, so static pages are available in every build variant including `make html-noplot`. That flag only skips executing the `plot_*.py` gallery scripts; it does not skip the page copy.

## Adding an Environment to the Docs

Every environment page is generated from the environment's own `README.md` using a Sphinx `{include}` directive. There are three steps:

### 1. Write the environment README

Your environment must have a `README.md` at `envs/<name>/README.md`. This file is the single source of truth — it renders on GitHub and is pulled into the docs site at build time.

Include HuggingFace frontmatter at the top, followed by a `# Title` heading (this becomes the page title and left nav label):

```markdown
---
title: My Environment
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# My Environment

Description, quick start, action/observation docs, etc.
```

### 2. Create the doc page

Create `docs/source/environments/<name>.md` with exactly this content:

````markdown
```{include} ../../../envs/<name>/README.md
```
````

This is the only pattern used — all 29 environment doc pages follow it. Do not add local headings or other content.

### 3. Add a card and toctree entry

Edit `docs/source/environments.md` to add two things:

**A card** inside the existing `{grid}` block (place alphabetically):

````markdown
````{grid-item-card} My Environment
:class-card: sd-border-1

Short one-line description of the environment.

+++
```{button-link} environments/<name>.html
:color: primary
:outline:

{octicon}`file;1em` Docs
```
```{button-link} https://huggingface.co/spaces/<org>/<name>
:color: warning
:outline:

🤗 Hugging Face
```
````
````

The Hugging Face button is optional — omit it if the environment isn't deployed to a Space.

**A toctree entry** in the `{toctree}` block at the bottom of the file (place alphabetically):

```
environments/<name>
```

### Verify

Rebuild and check that the environment appears in the left nav and the catalog grid:

```bash
cd docs && make clean html
cd _build/html && python -m http.server 8000
```
