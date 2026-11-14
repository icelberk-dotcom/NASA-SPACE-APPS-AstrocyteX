#!/usr/bin/env bash
# Publish AstroGuard-X to GitHub.
#   bash scripts/publish_github.sh [repo-name] [--private]
#   REMOTE_URL=https://github.com/icelberk-dotcom/NASA-space-apps-Ondankoymacano.git bash scripts/publish_github.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${1:-NASA-space-apps-Ondankoymacano}"; VIS="--public"; [ "${2:-}" = "--private" ] && VIS="--private"

command -v git >/dev/null || { echo "git is not installed"; exit 1; }
[ -d .git ] || git init -b main >/dev/null
git add -A

# Safety: never commit or publish secrets.
git ls-files | grep -qx ".env" && { echo ".env is tracked - run: git rm --cached .env"; exit 1; }
if git grep --cached -I -n -E '(AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_-]{20,})' >/dev/null 2>&1; then echo "Possible Google API key in staged/tracked files - aborting"; exit 1; fi

git diff --cached --quiet || git commit -m "Update AstroGuard-X" >/dev/null

if [ -n "${REMOTE_URL:-}" ]; then
  git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
  git push -u origin main
elif command -v gh >/dev/null; then
  gh repo create "$REPO" $VIS --source . --remote origin --push \
     --description "AI-powered astronaut health monitoring with gamified mitigation tasks (NASA Space Apps Challenge)"
else
  echo "GitHub CLI not found. Create an EMPTY repo at https://github.com/new, then run:"
  echo "  REMOTE_URL=https://github.com/<you>/$REPO.git bash scripts/publish_github.sh"
  exit 1
fi
echo "Published. Add topics (nasa-space-apps, astronaut-health, pytorch, mamba, streamlit) in the repo settings."
