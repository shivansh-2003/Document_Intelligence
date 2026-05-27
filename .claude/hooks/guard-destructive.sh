#!/usr/bin/env bash
# PreToolUse guard for Bash calls.
# Blocks destructive patterns and warns on risky operations.
# Exit 2 = block the tool call with message. Exit 0 = allow.

INPUT="${1:-}"

BLOCKED_PATTERNS=(
  "rm -rf /"
  "rm -rf \."
  "git push --force"
  "git reset --hard"
  "DROP TABLE"
  "TRUNCATE"
  "DELETE FROM.*WHERE.*1=1"
  "alembic downgrade base"
)

for pattern in "${BLOCKED_PATTERNS[@]}"; do
  if echo "$INPUT" | grep -qiE "$pattern"; then
    echo "BLOCKED: Destructive command pattern detected: '$pattern'" >&2
    echo "If this is intentional, the user must run it manually." >&2
    exit 2
  fi
done

exit 0
