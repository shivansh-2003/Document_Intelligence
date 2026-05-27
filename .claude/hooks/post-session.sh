#!/usr/bin/env bash
# Post-session stop hook.
# Runs when Claude Code ends a session. Reflects on what was learned
# and proposes targeted CLAUDE.md updates while context is fresh.

set -euo pipefail

SESSION_LOG="${CLAUDE_SESSION_LOG:-}"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M")
INSIGHTS_FILE=".claude/session-insights.md"

echo ""
echo "=== Post-Session Reflection ==="
echo "Timestamp: $TIMESTAMP"
echo ""

# Append a reflection prompt reminder to a running insights file.
# Claude reads this on the next session start (via pre-session.sh) to know
# what was learned last time.
cat >> "$INSIGHTS_FILE" <<EOF

---
## Session: $TIMESTAMP

<!-- Claude: after each session, append a 3-bullet summary here:
     - What you changed and why
     - Any pattern or gotcha discovered that isn't in CLAUDE.md
     - Any CLAUDE.md section that became stale / should be updated
     Keep bullets short — this file is read at session start. -->

EOF

echo "Insight stub written to $INSIGHTS_FILE"
echo "Review and fill in the session summary above to keep CLAUDE.md current."
echo ""

# Remind about CLAUDE.md review cadence if it has been long since last review.
LAST_REVIEW=$(git log --follow -1 --format="%ai" -- CLAUDE.md 2>/dev/null | cut -d' ' -f1 || echo "unknown")
echo "CLAUDE.md last changed: $LAST_REVIEW"
echo "Tip: review CLAUDE.md every 3-6 months or after major model/framework upgrades."
