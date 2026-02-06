# OpenEnv Documentation Reorganization Proposal

## Philosophy

Following the MCP and TRL documentation patterns, this proposal shifts OpenEnv docs from a "reference-first" to a "content-centric" approach. For a young project, users need:

1. **Quick wins** - Get something working in 5 minutes
2. **Conceptual understanding** - Why does OpenEnv exist? How does it work?
3. **Practical guidance** - How do I do X with OpenEnv?
4. **Reference** - API details when I need them

---

## Proposed Navigation Structure

```yaml
nav:
  - Get Started:
    - Introduction: introduction.md           # NEW - What is OpenEnv? Why use it?
    - Quick Start: quickstart.md              # EXISTING - 5-min hello world
    - Installation: installation.md           # NEW - Detailed installation options
    - Core Concepts: concepts.md              # NEW - Key abstractions explained

  - Guides:
    - Using Environments:
      - Auto-Discovery (AutoEnv): guides/auto-discovery.md    # MOVED from auto-discovery.md
      - Connecting to Servers: guides/connecting.md           # NEW - HTTP, Docker, HF Spaces
      - Async vs Sync: guides/async-sync.md                   # NEW - Extract from quickstart
    - Building Environments:
      - Your First Environment: guides/first-environment.md   # REFACTOR from environment-builder.md
      - Environment Anatomy: guides/environment-anatomy.md    # NEW - Deep dive into structure
      - Deployment Options: guides/deployment.md              # NEW - Docker, HF Spaces, registries
      - Web Interface: guides/web-interface.md                # NEW - Extract from README
    - Training Integration:
      - Using with RL Frameworks: guides/rl-integration.md    # NEW - Overview of integrations
      - Reward Design: guides/rewards.md                      # NEW - How to design reward functions

  - Tutorials:
    - Beginner:
      - OpenEnv in 15 Minutes: tutorials/openenv-tutorial.md  # EXISTING
      - Playing Blackjack with RL: tutorials/blackjack.md     # NEW - from examples/grpo_blackjack
    - Intermediate:
      - Wordle with GRPO & TRL: tutorials/wordle-grpo.md      # EXISTING
      - Training a Chess Agent: tutorials/chess-training.md   # NEW
    - Advanced:
      - Building a Custom Game Env: tutorials/custom-game.md  # NEW
      - Web Agents with BrowserGym: tutorials/web-agents.md   # NEW

  - Environments:
    - Overview: environments/index.md                          # REFACTOR from environments.md
    - By Category:
      - Games & Puzzles:
        - Echo (Testing): environments/echo.md
        - Snake: environments/snake.md
        - Chess: environments/chess.md
        - Atari: environments/atari.md
        - OpenSpiel: environments/openspiel.md
        - TextArena: environments/textarena.md
      - Code Execution:
        - Coding: environments/coding.md
        - REPL: environments/repl.md
        - Git: environments/git.md
        - Terminal-Bench 2: environments/tbench2.md
      - Web & Browser:
        - BrowserGym: environments/browsergym.md
        - OpenApp: environments/openapp.md
        - Web Search: environments/websearch.md
      - Simulation:
        - Unity ML-Agents: environments/unity.md
        - SUMO Traffic: environments/sumo.md
        - FinRL Trading: environments/finrl.md
      - Conversational:
        - Chat: environments/chat.md
        - DIPG Safety: environments/dipg.md

  - API Reference:
    - Python API:
      - Core: reference/core.md                # MOVED from core.md
      - Client Types: reference/client-types.md # NEW - EnvClient, StepResult, etc.
      - Server Types: reference/server-types.md # NEW - Environment, Action, Observation
    - CLI Reference: reference/cli.md          # MOVED from cli.md
    - Configuration:
      - openenv.yaml: reference/openenv-yaml.md # NEW - Manifest format reference
      - Environment Variables: reference/env-vars.md # NEW

  - Community:
    - Contributing: contributing.md             # LINK to CONTRIBUTING.md
    - RFCs: rfcs.md                            # NEW - Link to rfcs/
    - Partners & Supporters: partners.md       # NEW - Extract from README
```

---

## New Content Needed

### Priority 1: Core Conceptual Content (Write First)

| Page | Purpose | Source |
|------|---------|--------|
| `introduction.md` | What is OpenEnv? Problem it solves, key value props | Extract from index.md + README.md |
| `concepts.md` | Explain Environment, Action, Observation, State, StepResult | New content, reference existing diagrams |
| `installation.md` | pip, uv, Docker, from source | Extract from README + quickstart |

### Priority 2: Guides (Content-Centric How-Tos)

| Page | Purpose | Source |
|------|---------|--------|
| `guides/connecting.md` | How to connect to envs (URL, Docker, HF) | Extract from quickstart |
| `guides/async-sync.md` | When to use async vs sync, patterns | Extract from quickstart + env READMEs |
| `guides/first-environment.md` | Build your first env (simplified) | Refactor environment-builder.md |
| `guides/environment-anatomy.md` | Deep dive: models, server, client, Dockerfile | Refactor environment-builder.md |
| `guides/deployment.md` | Deploy to HF Spaces, Docker registries | Extract from environment-builder + CLI |
| `guides/rl-integration.md` | Using with TRL, torchforge, SkyRL, etc. | Extract from README partner section |
| `guides/rewards.md` | Designing reward functions | New, examples from dipg, chess, textarena |

### Priority 3: More Tutorials

| Page | Purpose | Source |
|------|---------|--------|
| `tutorials/blackjack.md` | Complete GRPO example | From examples/grpo_blackjack |
| `tutorials/chess-training.md` | Train a chess agent | New, use chess_env |
| `tutorials/web-agents.md` | Build web agent with BrowserGym | New, from browsergym_env |

---

## Content Migration Map

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `index.md` | `introduction.md` | Refactor into proper intro |
| `quickstart.md` | `quickstart.md` | Keep, extract async/sync to guide |
| `auto-discovery.md` | `guides/auto-discovery.md` | Move |
| `environment-builder.md` | `guides/first-environment.md` + `guides/environment-anatomy.md` | Split |
| `environments.md` | `environments/index.md` | Move, add categories |
| `tutorials/openenv-tutorial.md` | `tutorials/openenv-tutorial.md` | Keep |
| `tutorials/wordle-grpo.md` | `tutorials/wordle-grpo.md` | Keep, fix backtick escaping |
| `core.md` | `reference/core.md` | Move |
| `cli.md` | `reference/cli.md` | Move |

---

## Proposed mkdocs.yml

```yaml
site_name: OpenEnv
site_description: Agentic execution environments for RL training
site_url: https://meta-pytorch.github.io/OpenEnv
repo_url: https://github.com/meta-pytorch/OpenEnv
repo_name: meta-pytorch/OpenEnv

docs_dir: .
site_dir: ../site

theme:
  name: material
  font:
    text: Inter
  logo: https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609
  palette:
    - scheme: slate
      primary: grey
      accent: blue grey
  features:
    - navigation.sections      # Group nav items
    - navigation.expand
    - navigation.top
    - navigation.indexes       # Allow index pages for sections
    - toc.follow
    - content.code.copy
    - content.tabs.link        # Link code tabs across page
    - search.highlight
    - search.suggest

extra_css:
  - styles/theme.css

plugins:
  - search
  - include-markdown:
      rewrite_relative_urls: true
      opening_tag: "--8<--"
      closing_tag: ""
  - mkdocstrings:
      handlers:
        python:
          paths:
            - ../src
          options:
            show_source: false
            members_order: source

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - def_list
  - footnotes
  - tables
  - toc:
      permalink: true
      toc_depth: 2
  - pymdownx.blocks.admonition:
      types:
        - name: note
        - name: tip
        - name: important
        - name: warning
        - name: caution
  - meta

nav:
  - Get Started:
    - Introduction: introduction.md
    - Quick Start: quickstart.md
    - Installation: installation.md
    - Core Concepts: concepts.md

  - Guides:
    - guides/index.md
    - Using Environments:
      - Auto-Discovery: guides/auto-discovery.md
      - Connecting to Servers: guides/connecting.md
      - Async vs Sync Usage: guides/async-sync.md
    - Building Environments:
      - Your First Environment: guides/first-environment.md
      - Environment Anatomy: guides/environment-anatomy.md
      - Deployment: guides/deployment.md
    - Training:
      - RL Framework Integration: guides/rl-integration.md
      - Reward Design: guides/rewards.md

  - Tutorials:
    - tutorials/index.md
    - OpenEnv in 15 Minutes: tutorials/openenv-tutorial.md
    - Wordle with GRPO: tutorials/wordle-grpo.md
    - Blackjack Training: tutorials/blackjack.md

  - Environments:
    - environments/index.md
    - Games:
      - Echo: environments/echo.md
      - Snake: environments/snake.md
      - Chess: environments/chess.md
      - Atari: environments/atari.md
      - OpenSpiel: environments/openspiel.md
      - TextArena: environments/textarena.md
    - Code Execution:
      - Coding: environments/coding.md
      - REPL: environments/repl.md
      - Git: environments/git.md
      - Terminal-Bench 2: environments/tbench2.md
    - Web & Browser:
      - BrowserGym: environments/browsergym.md
      - OpenApp: environments/openapp.md
      - Web Search: environments/websearch.md
    - Simulation:
      - Unity: environments/unity.md
      - SUMO Traffic: environments/sumo.md
      - FinRL: environments/finrl.md
    - Conversational:
      - Chat: environments/chat.md
      - DIPG Safety: environments/dipg.md

  - API Reference:
    - reference/index.md
    - Core API: reference/core.md
    - CLI: reference/cli.md
    - openenv.yaml: reference/openenv-yaml.md

  - Community:
    - Contributing: contributing.md
    - Partners: partners.md
```

---

## Visual Comparison

### Current Structure (Reference-Heavy)
```
Get Started
├── What is OpenEnv?      (landing page)
├── Quick Start           (tutorial-ish)
├── Auto-Discovery API    (reference)
└── Building an Env       (long tutorial)

Tutorials                  (only 2)

Environments              (flat list, no categories)

API Reference
├── Core
└── CLI
```

### Proposed Structure (Content-Centric)
```
Get Started
├── Introduction          (why OpenEnv?)
├── Quick Start           (5-min win)
├── Installation          (all options)
└── Core Concepts         (understand the model)

Guides                     (how to do X)
├── Using Environments
│   ├── Auto-Discovery
│   ├── Connecting
│   └── Async vs Sync
├── Building Environments
│   ├── First Environment
│   ├── Anatomy
│   └── Deployment
└── Training
    ├── RL Integration
    └── Reward Design

Tutorials                  (end-to-end walkthroughs)
├── OpenEnv in 15 Minutes
├── Wordle with GRPO
└── Blackjack Training

Environments               (categorized)
├── Games & Puzzles (6)
├── Code Execution (4)
├── Web & Browser (3)
├── Simulation (3)
└── Conversational (2)

API Reference              (when you need details)
├── Core API
├── CLI
└── Config Files

Community
├── Contributing
└── Partners
```

---

## Implementation Phases

### Phase 1: Restructure (Week 1)
1. Create new folder structure (`guides/`, `reference/`, `environments/`)
2. Move existing files to new locations
3. Update mkdocs.yml with new nav
4. Fix all internal links

### Phase 2: Core Content (Week 2)
1. Write `introduction.md` - clear value proposition
2. Write `concepts.md` - Environment, Action, Observation explained
3. Write `installation.md` - consolidate installation info
4. Split `environment-builder.md` into guides

### Phase 3: Guides (Week 3)
1. Write `guides/connecting.md`
2. Write `guides/async-sync.md`
3. Write `guides/rl-integration.md`
4. Write `guides/deployment.md`

### Phase 4: Polish (Week 4)
1. Add more tutorials
2. Categorize environments properly
3. Add metadata (authors, dates) to all pages
4. Review and fix formatting issues from audit

---

## Key Principles (from MCP/TRL patterns)

1. **Progressive Disclosure**: Start simple, go deep as needed
2. **Task-Oriented Guides**: "How do I..." not "Here's the API for..."
3. **Clear Categories**: Games, Code, Web, Simulation - not a flat list
4. **Conceptual Foundation**: Explain the model before the API
5. **Real Examples**: Every guide should have runnable code
6. **Cross-Linking**: Reference docs link to guides, guides link to tutorials

---

## Questions to Resolve

1. Should environment docs live in `/docs/environments/` or continue using includes from `/envs/*/README.md`?
   - **Recommendation**: Keep includes, but add intro content to each page

2. Should we have a separate "Cookbook" section for recipes?
   - **Recommendation**: Not yet - fold into Guides for now

3. How to handle partner integrations (TRL, torchforge, SkyRL)?
   - **Recommendation**: One comprehensive "RL Integration" guide + link to partner docs
