---
name: rfc-check
description: Determine if proposed changes require an RFC. Use when planning significant changes, before starting major work, or when asked whether an RFC is needed.
allowed-tools: Read, Grep, Glob
---

# RFC Check

Determine if proposed changes require an RFC (Request for Comments).

## Instructions

1. **Identify changed files** using `git diff --name-only` or provided context

2. **Apply RFC criteria**:

   **RFC Required**:
   - New APIs in `src/openenv/core/`
   - Breaking changes to existing APIs
   - New abstractions or design patterns
   - Changes affecting the two-interface model (WebSocket/MCP separation)
   - Major architectural decisions

   **RFC Not Required**:
   - Bug fixes
   - Documentation updates
   - Minor refactoring (no API changes)
   - New example environments (unless introducing new patterns)
   - Dependency updates
   - Test additions

3. **Check against existing RFCs** in `rfcs/` for conflicts or dependencies

## Analysis Steps

1. List all files being changed
2. Identify any files in `src/openenv/core/`
3. Check for public API signature changes
4. Look for new abstractions or patterns
5. Review existing RFCs for related decisions

## Output Format

```
## RFC Analysis

### Files Changed
- [list of files]

### Core Files Touched
- [any files in src/openenv/core/, or "None"]

### API Changes
- [any signature changes to public APIs, or "None"]

### New Patterns/Abstractions
- [any new patterns introduced, or "None"]

### Verdict: NOT REQUIRED / RECOMMENDED / REQUIRED

### Reasoning
[Explanation of decision based on criteria above]

### If RFC Needed
- Suggested title: "RFC NNN: [title]"
- Related RFCs: [list any related existing RFCs]
- Key decisions to document: [list]
```

## RFC Template Reference

If an RFC is needed, use the template in `rfcs/README.md`:

```markdown
# RFC NNN: Title

**Status**: Draft
**Created**: YYYY-MM-DD
**Authors**: @username

## Summary
[1-2 paragraph overview]

## Motivation
[Problem Statement + Goals]

## Design
[Architecture Overview, Core Abstractions, Key Design Decisions]

## Examples
[Code samples demonstrating usage]
```
