#!/usr/bin/env bash
# Run once with a GitHub account/token that has repository administration access.
# Requires gh authentication. The CI workflow must have run on main first.
set -euo pipefail

repository="${1:-jekkiechan/more-than-a-hack-2027}"

if gh api "repos/${repository}/branches/main/protection" >/dev/null 2>&1; then
  echo "main already has branch protection; review it before replacing settings." >&2
  exit 1
fi

gh api --method PUT "repos/${repository}/branches/main/protection" --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "contexts": ["tests"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON

gh api "repos/${repository}/branches/main/protection" \
  --jq '{required_status_checks, enforce_admins, required_pull_request_reviews, required_linear_history, allow_force_pushes, allow_deletions}'
