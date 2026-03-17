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
