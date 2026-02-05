---
name: alignment-review
description: Review code changes for bugs and alignment with OpenEnv principles and RFCs. Use when reviewing PRs, checking code before commit, or when asked to review changes. Implements two-tier review model.
allowed-tools: Read, Grep, Glob, Bash
---

# Alignment Review

Review code changes for alignment with OpenEnv principles using a two-tier model.

## Instructions

1. **Run automated checks first**:
   - Execute `bash .claude/hooks/lint.sh` - capture lint issues
   - Execute `bash .claude/hooks/check-debug.sh` - capture debug code

2. **Read alignment documents**:
   - `.claude/docs/PRINCIPLES.md` - design principles
   - `.claude/docs/INVARIANTS.md` - system invariants

3. **Read open RFCs**:
   - Scan `rfcs/` directory for all RFC files
   - Note the status of each RFC (Draft, In Review, Accepted, Implemented)
   - Pay special attention to Draft and In Review RFCs - these represent active design discussions

4. **Analyze changes** (use `git diff` or provided diff):
   - Identify mechanical issues (Tier 1)
   - Flag alignment concerns (Tier 2)
   - Flag conflicts with open RFCs (Tier 2)

## Tier 1: Uncontentious Issues (Fix Immediately)

These are issues to fix without human input:
- Lint failures from hook output
- Debug code from hook output (print statements, breakpoints)
- Uninitialized variables, type errors
- Missing imports, syntax errors
- Security issues (credential exposure, injection vulnerabilities)

## Tier 2: Alignment Discussion Points

For each potential alignment concern, format as:

```
**ALIGNMENT FLAG**: [Brief description]
- **Principle/RFC at stake**: [Which principle from PRINCIPLES.md or RFC number]
- **The concern**: [What seems misaligned or in conflict]
- **Suggested reviewer**: @darktex [pull actual reviewers based on authors of the specific line of PRINCIPLES.md and INVARIANTS.md using git blame, and/or authors of conflicting RFCs]
```

### Examples of Tier 2 Issues

**Principle conflicts:**
- Adding external reward computation (violates "rewards in environment")
- Client importing server code (violates client-server separation)
- New API that differs from Gymnasium pattern

**RFC conflicts (flag even for Draft/In Review RFCs):**
- Change conflicts with design proposed in an open RFC
- Change pre-empts a decision being discussed in an RFC
- Change implements something differently than an RFC proposes
- Change affects an area covered by an RFC under review

**Why flag RFC conflicts?** Even if an RFC isn't finalized, flagging conflicts helps focus design discussions. The change might be correct and the RFC might need updating, or vice versa - either way, the team should discuss.

## Output Format

```
## Alignment Review Report

### Automated Checks
- Lint: [PASS/FAIL] - [summary]
- Debug code: [CLEAN/FOUND] - [details]

### Open RFCs Context
[List any RFCs in Draft or In Review status that might be relevant to these changes]

### Tier 1: Fixes Required
- [ ] path/file.py:123 - [issue description]
- [ ] path/file.py:456 - [issue description]

### Tier 2: Alignment Discussion

#### Principle Conflicts
[ALIGNMENT FLAGS for principle violations, or "None identified"]

#### RFC Conflicts
[ALIGNMENT FLAGS for RFC conflicts, or "None identified"]

### Summary
- X mechanical issues to fix
- Y alignment points for human review
- Z RFC conflicts to discuss
```
