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
# Install git hooks for the team
bash .claude/hooks/install.sh
# Run Claude Code - skills and agents are automatically available
```

Verify with `/agents` or ask "what skills are available?"

## Two Development Modes

OpenEnv supports two development modes based on your location:

### Explore Mode (Main Repo)

When working in the main repository clone, direct edits are allowed:
- Quick exploration and prototyping
- Small fixes that don't need TDD workflow
- Documentation updates

### TDD Mode (Worktrees)

When working in a worktree (`.worktrees/<name>/`), TDD is enforced:
- Direct code edits are blocked
- Must use `/write-tests` → `/implement` workflow
- Say "skip TDD" to bypass blocking

### Creating a Worktree

```bash
.claude/scripts/worktree-create.sh add-feature
cd .worktrees/add-feature
# Now in TDD mode
```

### TDD Workflow

```
/work-on-issue #42  →  Start from GitHub issue
    ↓
/write-tests        →  Create failing tests (Red)
    ↓
/implement          →  Make tests pass (Green)
    ↓
/simplify           →  Refactor (optional)
    ↓
/pre-submit-pr      →  Validate before PR
```

### Skills vs Agents

- **Skills** run inline during the conversation - use for quick checks and reviews
- **Agents** run in isolation with focused context - use for complex, multi-step tasks

### Available Skills

Skills are defined in `.claude/skills/` and run inline:

**Review & Validation Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`alignment-review`](.claude/skills/alignment-review/SKILL.md) | "review this code" | Two-tier review (bugs + alignment flags) |
| [`pre-submit-pr`](.claude/skills/pre-submit-pr/SKILL.md) | "ready for PR?" | Comprehensive PR readiness check |
| [`rfc-check`](.claude/skills/rfc-check/SKILL.md) | "do I need an RFC?" | Determine if RFC required |

**TDD Workflow Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`work-on-issue`](.claude/skills/work-on-issue/SKILL.md) | "/work-on-issue #42" | Start TDD workflow from GitHub issue |
| [`write-tests`](.claude/skills/write-tests/SKILL.md) | "/write-tests" | Write failing tests (Red phase) |
| [`implement`](.claude/skills/implement/SKILL.md) | "/implement" | Make tests pass (Green phase) |
| [`simplify`](.claude/skills/simplify/SKILL.md) | "/simplify" | Refactor after tests pass |

### Available Subagents

Agents are defined in `.claude/agents/` and run in isolation:

**Review & Validation Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `alignment-reviewer` | Review PRs for bugs + alignment | [.claude/agents/alignment-reviewer.md](.claude/agents/alignment-reviewer.md) |
| `env-validator` | Validate environments end-to-end | [.claude/agents/env-validator.md](.claude/agents/env-validator.md) |
| `openenv-architect` | Design new features/environments | [.claude/agents/openenv-architect.md](.claude/agents/openenv-architect.md) |
| `build-validator` | Validate builds before merge | [.claude/agents/build-validator.md](.claude/agents/build-validator.md) |

**TDD Workflow Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `issue-worker` | Extract requirements from GitHub issues | [.claude/agents/issue-worker.md](.claude/agents/issue-worker.md) |
| `pr-planner` | Plan stacked PRs for complex features | [.claude/agents/pr-planner.md](.claude/agents/pr-planner.md) |
| `tester` | Write high-signal, failing tests | [.claude/agents/tester.md](.claude/agents/tester.md) |
| `implementer` | Make tests pass with minimal code | [.claude/agents/implementer.md](.claude/agents/implementer.md) |

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
| [TESTING_STRATEGY.md](.claude/docs/TESTING_STRATEGY.md) | Testing philosophy and patterns |
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

# Lint check (format + rules)
uv run ruff format src/ tests/ --check
uv run ruff check src/ tests/

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

## Git Hooks

Install git hooks for team-wide consistency:

```bash
bash .claude/hooks/install.sh
```

This installs:
- **pre-commit**: Branch check (blocks main), format, lint, debug artifacts
- **commit-msg**: Issue reference reminder (soft warning)
- **pre-push**: Format, lint, tests, invariants, conflict detection
- **post-merge**: Worktree cleanup reminder

Skip temporarily with `git commit/push --no-verify`.

## Worktree Management

For focused feature work, use worktrees:

```bash
# Create a worktree for a feature
.claude/scripts/worktree-create.sh add-mcp-tools
cd .worktrees/add-mcp-tools

# When done, clean up
.claude/scripts/worktree-cleanup.sh .worktrees/add-mcp-tools
```

Worktrees enable:
- Isolated branches without switching
- TDD enforcement via hooks
- Parallel work on multiple features
