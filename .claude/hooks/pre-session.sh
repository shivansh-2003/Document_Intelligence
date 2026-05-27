#!/usr/bin/env bash
# Pre-session start hook.
# Runs when Claude Code begins a session. Surfaces recent insights and
# any pending CLAUDE.md update proposals so the model starts with full context.

set -euo pipefail

INSIGHTS_FILE=".claude/session-insights.md"
PENDING_UPDATES=".claude/pending-claude-md-updates.md"

echo ""
echo "=== Document Intelligence — Session Start ==="
echo "Date: $(date '+%Y-%m-%d %H:%M')"
echo ""

# Surface last session's insights if they exist.
if [[ -f "$INSIGHTS_FILE" ]]; then
  LAST_ENTRY=$(awk '/^---/{found=1; content=""} found{content=content"\n"$0} END{print content}' "$INSIGHTS_FILE" | tail -20)
  if [[ -n "$LAST_ENTRY" ]]; then
    echo "--- Recent session insights ---"
    echo "$LAST_ENTRY"
    echo ""
  fi
fi

# Surface any pending CLAUDE.md update proposals.
if [[ -f "$PENDING_UPDATES" ]]; then
  echo "--- Pending CLAUDE.md updates (review and apply if accurate) ---"
  cat "$PENDING_UPDATES"
  echo ""
fi

# Print active branch and uncommitted changes summary.
if git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
  BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
  DIRTY=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
  echo "Git: branch=$BRANCH  uncommitted-files=$DIRTY"
  echo ""
fi

echo "Architecture: context/architecture.md"
echo "Folder map:   context/folder.md"
echo "============================================="
echo ""
