# Contributing with Claude Code

OpenEnv is an agentic-first project. We expect most contributions to use Claude Code or similar tools. This document describes the workflow.

## The Two-Phase Model

### Phase 1: Design & Alignment (Human-Owned)

Humans own the "what" and "why":
- Major architectural decisions require RFCs
- Discuss trade-offs in issues before implementation
- Establish acceptance criteria and invariants
- Review for alignment, not just correctness

### Phase 2: Implementation (Claude-Owned)

Claude handles the mechanical loop:
```
while not working:
    try_some_shit()
    test()
```

Humans intervene only for alignment questions.

## When to Write an RFC

**Required for:**
- New core APIs in `src/openenv/core/`
- Breaking changes to existing APIs
- Major architectural decisions
- New abstractions or design patterns
- Changes affecting the two-interface model (WebSocket/MCP)

**Not required for:**
- Bug fixes, documentation, minor refactoring
- New example environments (unless introducing new patterns)
- Dependency updates, test additions

See `rfcs/README.md` for the RFC process.

## Review Expectations

### What Claude Catches (Tier 1)
- Bugs, uninitialized variables, type errors
- Lint failures, test failures
- Security issues (credential exposure, injection)
- Debug code left in (print statements, breakpoints)

### What Humans Review (Tier 2)
- Does this align with our principles in PRINCIPLES.md?
- Does this maintain our invariants in INVARIANTS.md?
- Is this the right trade-off for the project?
- Should this decision be documented in an RFC?

### Alignment Flags

When Claude identifies a potential alignment issue, it formats as:
```
**ALIGNMENT FLAG**: [Brief description]
- **Principle at stake**: [Which principle]
- **The concern**: [What seems misaligned]
- **Suggested reviewer**: @[maintainer]
```

## Available Tools

For the full list of available skills, subagents, and recommended plugins, see [CLAUDE.md](../../CLAUDE.md#available-skills).
