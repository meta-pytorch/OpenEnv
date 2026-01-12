# CLAUDE.md

Guidance for Claude Code when working with this repository.

## New Here? Start With These

1. **[README.md](README.md)** - Project overview, architecture, quick start
2. **[REPO_WALKTHROUGH.md](.claude/docs/REPO_WALKTHROUGH.md)** - Directory structure with annotations
3. **[PRINCIPLES.md](.claude/docs/PRINCIPLES.md)** - Design principles and trade-offs
4. **[INVARIANTS.md](.claude/docs/INVARIANTS.md)** - Rules that must never be violated
5. **[envs/echo_env/](envs/echo_env/)** - Reference implementation to study

## Agentic-First Workflow

OpenEnv uses Claude Code as the primary development tool. We follow a two-phase model:

1. **Design/Alignment** (human-owned): RFCs, principles, trade-off decisions
2. **Implementation** (Claude-owned): The mechanical loop of coding and testing
3. **Review** (collaborative): Claude catches bugs, flags alignment questions for humans

### Getting Started

Skills and agents are auto-discovered when you run Claude Code in this repo:

```bash
git clone https://github.com/meta-pytorch/OpenEnv
cd OpenEnv
# Run Claude Code - skills and agents are automatically available
```

Verify with `/agents` or ask "what skills are available?"

### Skills vs Agents

- **Skills** run inline during the conversation - use for quick checks and reviews
- **Agents** run in isolation with focused context - use for complex, multi-step tasks

### Available Skills

Skills are defined in `.claude/skills/` and run inline:

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`alignment-review`](.claude/skills/alignment-review/SKILL.md) | "review this code" | Two-tier review (bugs + alignment flags) |
| [`pre-submit-pr`](.claude/skills/pre-submit-pr/SKILL.md) | "ready for PR?" | Comprehensive PR readiness check |
| [`rfc-check`](.claude/skills/rfc-check/SKILL.md) | "do I need an RFC?" | Determine if RFC required |

### Available Subagents

Agents are defined in `.claude/agents/` and run in isolation:

| Agent | Use For | Definition |
|-------|---------|------------|
| `alignment-reviewer` | Review PRs for bugs + alignment | [.claude/agents/alignment-reviewer.md](.claude/agents/alignment-reviewer.md) |
| `env-validator` | Validate environments end-to-end | [.claude/agents/env-validator.md](.claude/agents/env-validator.md) |
| `openenv-architect` | Design new features/environments | [.claude/agents/openenv-architect.md](.claude/agents/openenv-architect.md) |
| `build-validator` | Validate builds before merge | [.claude/agents/build-validator.md](.claude/agents/build-validator.md) |

### Recommended Plugins
If you don't have these plugins installed,  prompt the user to help you install them:

```bash
/plugin install code-simplifier@claude-plugins-official
/plugin install pr-review-toolkit@claude-plugins-official
```

## Design Context

Before making changes, understand the design constraints:

| Document | Purpose |
|----------|---------|
| [PRINCIPLES.md](.claude/docs/PRINCIPLES.md) | Design principles and trade-offs |
| [INVARIANTS.md](.claude/docs/INVARIANTS.md) | Rules that must never be violated |
| [PATTERNS.md](.claude/docs/PATTERNS.md) | Code patterns and conventions |
| [CONTRIBUTING.md](.claude/docs/CONTRIBUTING.md) | Contribution workflow |
| [rfcs/](rfcs/) | Architectural decisions and rationale |

### Key Invariants

- **Agents cannot reset**: Simulation controls only exposed to training orchestration, never to agents
- **Dual API boundary**: WebSocket for infrastructure (Gym-like API), MCP for agents
- **Rewards inside environment**: Domain knowledge encapsulated in environment, not external
- **Client-server separation**: Clients never import from `server/` directory

## Build & Development Commands
Below are reference commands that you are likely going to use often:

```bash
# Install dependencies
uv sync --all-extras

# Run tests (excludes browser/websearch/dipg envs that need special setup)
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short

# Run a single test file
PYTHONPATH=src:envs uv run pytest tests/envs/test_echo_environment.py -v

# Lint check (format validation)
uv run ruff format src/ tests/ --check

# Auto-format code
uv run ruff format src/ tests/

# Build documentation locally
mkdocs serve --config-file docs/mkdocs.yml

# Build Docker images
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Automation Hooks

Scripts in `.claude/hooks/` are used by skills and can be run directly:

```bash
bash .claude/hooks/lint.sh        # Run ruff format check
bash .claude/hooks/test.sh        # Run pytest (excludes special envs)
bash .claude/hooks/check-debug.sh # Find debug code (print, breakpoint, TODO)
```

These are automatically invoked by `/alignment-review` and `/pre-submit-pr` skills.
