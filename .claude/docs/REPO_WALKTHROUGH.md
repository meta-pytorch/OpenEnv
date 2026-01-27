# Repository Walkthrough

This document provides a navigational guide to the OpenEnv codebase.

## Top-Level Structure

```
OpenEnv/
├── CLAUDE.md                 # Entry point for Claude Code - build commands, architecture overview
├── README.md                 # Project overview and getting started
├── pyproject.toml            # Python package configuration (uv/pip)
├── uv.lock                   # Locked dependencies
│
├── src/                      # Core library code (installed as `openenv`)
├── envs/                     # Example environments (not installed, used via PYTHONPATH)
├── tests/                    # Test suite
├── examples/                 # Usage examples and tutorials
├── docs/                     # Documentation (MkDocs)
├── rfcs/                     # Design documents and architectural decisions
├── scripts/                  # Utility scripts
│
├── .claude/                  # Claude Code configuration (skills, agents, docs)
├── .github/                  # GitHub Actions, PR templates, issue templates
└── .gitignore
```

## Source Code (`src/`)

```
src/
├── openenv/                  # Main package
│   ├── __init__.py
│   │
│   ├── core/                 # Core abstractions - the heart of OpenEnv
│   │   ├── env_client.py         # EnvClient base class (WebSocket client)
│   │   ├── client_types.py       # Client-side type definitions
│   │   ├── utils.py              # Shared utilities
│   │   │
│   │   ├── env_server/           # Server-side components
│   │   │   ├── interfaces.py         # Environment abstract base class
│   │   │   ├── http_server.py        # HTTPEnvServer (FastAPI + WebSocket)
│   │   │   ├── types.py              # Wire types (Action, Observation, State, WS messages)
│   │   │   ├── serialization.py      # Pydantic serialization helpers
│   │   │   ├── base_transforms.py    # Transform pipeline for rewards/observations
│   │   │   ├── web_interface.py      # Web UI for debugging environments
│   │   │   ├── route_config.py       # FastAPI route configuration
│   │   │   └── exceptions.py         # Server-side exceptions
│   │   │
│   │   ├── containers/           # Container lifecycle management
│   │   │   ├── runtime/              # Provider implementations
│   │   │   │   ├── local_docker.py       # LocalDockerProvider
│   │   │   │   └── uv_provider.py        # UVProvider (for local dev)
│   │   │   └── images/               # Base Docker images
│   │   │       └── Dockerfile            # openenv-base image
│   │   │
│   │   └── tools/                # Reusable tool implementations
│   │       ├── local_python_executor.py  # Python code execution
│   │       └── git_server_client.py      # Git operations
│   │
│   └── cli/                  # Command-line interface
│       ├── __main__.py           # Entry point (`python -m openenv.cli`)
│       ├── commands/             # CLI subcommands
│       │   ├── init.py               # `openenv init` - scaffold new env
│       │   ├── serve.py              # `openenv serve` - run server locally
│       │   ├── build.py              # `openenv build` - build Docker image
│       │   ├── push.py               # `openenv push` - deploy to HF Spaces
│       │   └── validate.py           # `openenv validate` - check config
│       └── templates/            # Scaffolding templates
│           └── openenv_env/          # Template for `openenv init`
│
└── openenv_core/             # Legacy compatibility shim (imports from openenv.core)
```

## Environments (`envs/`)

Each environment follows a consistent structure:

```
envs/
├── echo_env/                 # Minimal reference environment
│   ├── client.py                 # EnvClient subclass
│   ├── models.py                 # Action, Observation, State models
│   ├── openenv.yaml              # Environment manifest
│   ├── pyproject.toml            # Environment-specific dependencies
│   ├── README.md
│   └── server/
│       ├── app.py                    # FastAPI app setup
│       ├── echo_environment.py       # Environment implementation
│       └── Dockerfile                # Container definition
│
├── coding_env/               # Python code execution environment
├── chat_env/                 # Conversational environment
├── textarena_env/            # Text-based games (TextArena)
├── browsergym_env/           # Browser automation (BrowserGym)
├── openspiel_env/            # Game theory environments (OpenSpiel)
├── atari_env/                # Atari games via Gymnasium
├── finrl_env/                # Financial RL environment
├── git_env/                  # Git operations environment
├── snake_env/                # Classic Snake game
├── sumo_rl_env/              # Traffic simulation (SUMO)
├── connect4_env/             # Connect Four game
├── dipg_safety_env/          # Safety-focused environment
├── reasoning_gym_env/        # Reasoning problems and puzzles
└── websearch_env/            # Web search environment
```

## Tests (`tests/`)

```
tests/
├── conftest.py               # Pytest fixtures
├── test_*.py                 # Core library tests
│
├── envs/                     # Per-environment integration tests
│   ├── test_echo_environment.py
│   ├── test_coding_environment.py
│   └── ...
│
├── test_cli/                 # CLI command tests
└── scripts/                  # Test utility scripts
```

## RFCs (`rfcs/`)

Design documents that capture architectural decisions:

```
rfcs/
├── README.md                 # RFC process and template
├── 000-project-phases.md     # Project vision and phases
├── 001-abstractions.md       # Core abstractions (Environment, Client, two-interface model)
├── 002-env-spec.md           # Environment specification
└── 003-mcp-support.md        # MCP integration design
```

## Claude Code Configuration (`.claude/`)

```
.claude/
├── docs/                     # Alignment documents
│   ├── PRINCIPLES.md             # Design principles and trade-offs
│   ├── INVARIANTS.md             # System invariants (must never violate)
│   ├── PATTERNS.md               # Code patterns and conventions
│   ├── CONTRIBUTING.md           # Agentic contribution workflow
│   └── REPO_WALKTHROUGH.md       # This file
│
├── skills/                   # Auto-discovered skills
│   ├── alignment-review/
│   │   └── SKILL.md              # Two-tier code review
│   ├── pre-submit-pr/
│   │   └── SKILL.md              # PR readiness validation
│   └── rfc-check/
│       └── SKILL.md              # RFC requirement analysis
│
├── agents/                   # Specialized subagents
│   ├── alignment-reviewer.md     # Review for bugs + alignment
│   ├── env-validator.md          # Validate environments e2e
│   ├── openenv-architect.md      # Design new features
│   └── build-validator.md        # Validate builds
│
└── hooks/                    # Automation scripts
    ├── lint.sh                   # Run ruff format check
    ├── test.sh                   # Run pytest
    └── check-debug.sh            # Find debug code
```

## Documentation (`docs/`)

MkDocs-based documentation:

```
docs/
├── mkdocs.yml                # MkDocs configuration
├── index.md                  # Home page
├── quickstart.md             # Getting started guide
├── core.md                   # Core library documentation
├── cli.md                    # CLI reference
├── environment-builder.md    # How to create environments
│
├── environments/             # Per-environment documentation
│   ├── echo.md
│   ├── coding.md
│   └── ...
│
├── tutorials/                # Step-by-step guides
│   ├── openenv-tutorial.md
│   └── wordle-grpo.md
│
└── styles/                   # Custom CSS
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/openenv/core/env_server/interfaces.py` | `Environment` abstract base class |
| `src/openenv/core/env_client.py` | `EnvClient` WebSocket client |
| `src/openenv/core/env_server/http_server.py` | `HTTPEnvServer` FastAPI wrapper |
| `src/openenv/core/env_server/types.py` | All wire types and WebSocket messages |
| `envs/echo_env/` | Reference implementation - start here |
| `rfcs/001-abstractions.md` | Core architectural decisions |
| `.claude/docs/INVARIANTS.md` | Rules that must never be broken |
