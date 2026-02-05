# PR Review Bot

Automated PR review system using Claude Code. Runs as a cron job to review open PRs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cron (every 6 hours)                     │
│                pr-review-cron-wrapper.sh                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                            │
│  1. Calls pr_tracker.py --list --since 6h                   │
│  2. Spawns alignment-reviewer subagent for each PR          │
│  3. Posts reviews via pr_tracker.post_review()              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Claude handles orchestration. The only code needed is `pr_tracker.py` for GitHub API access.

## Files

| File | Purpose |
|------|---------|
| `pr_tracker.py` | GitHub API wrapper (PyGithub) - fetches PRs, posts reviews |
| `pr-review-cron-wrapper.sh` | Cron entry point - invokes Claude |

## Quick Start

### 1. Install PyGithub

```bash
pip install PyGithub
```

### 2. Test the Tracker

```bash
# List PRs updated in last 6 hours
python3 scripts/pr_tracker.py --list --since 6h

# List PRs updated in last day
python3 scripts/pr_tracker.py --list --since 1d

# Get details for a specific PR
python3 scripts/pr_tracker.py --details 123
```

### 3. Test a Review (manually)

```bash
claude "Review PR #123 using the alignment-reviewer agent.
Use scripts/pr_tracker.py to get PR details and post the review."
```

### 4. Set Up Cron

```bash
crontab -e

# Add this line
0 */6 * * * /home/davidet/OpenEnv/scripts/pr-review-cron-wrapper.sh >> ~/.openenv-review-cron.log 2>&1
```

## pr_tracker.py API

```python
from scripts.pr_tracker import (
    get_prs_needing_review,
    get_pr_details,
    post_review,
    parse_since,
)

# Get PRs updated in last 6 hours
since = parse_since("6h")
prs = get_prs_needing_review(since=since)
# Returns: [{"number": 123, "title": "...", "updated_at": "...", ...}]

# Get detailed info about a PR
details = get_pr_details(123)
# Returns: {"number": 123, "files": [...], "body": "...", ...}

# Post a review
post_review(pr_number=123, verdict="approve", body="LGTM!")
```

## Filtering

PRs are filtered by update time using `--since`:
- `6h` - last 6 hours
- `1d` - last day
- `2w` - last 2 weeks
- `2024-01-13T00:00:00Z` - since specific timestamp

The cron job runs every 6 hours with `--since 6h`, so it reviews any PR that was updated since the last run.

## Review Model

| Issues Found | Verdict |
|--------------|---------|
| Tier 1 issues (bugs, lint, security) | `request_changes` |
| Only Tier 2 flags (alignment concerns) | `comment` |
| No issues | `approve` |

## Logs

```bash
tail -50 ~/.openenv-review-cron.log
```
