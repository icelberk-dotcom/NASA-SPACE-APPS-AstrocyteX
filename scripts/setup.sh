#!/usr/bin/env bash
# One-shot setup for AstroGuard-X on Linux / macOS.
#   bash scripts/setup.sh [--skip-train] [--skip-tests] [--skip-git] [--gpu]
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_TRAIN=0; SKIP_TESTS=0; SKIP_GIT=0; GPU=0
for a in "$@"; do
  case "$a" in
    --skip-train) SKIP_TRAIN=1;; --skip-tests) SKIP_TESTS=1;; --skip-git) SKIP_GIT=1;; --gpu) GPU=1;;
    *) echo "unknown option $a"; exit 2;;
  esac
done
step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }

step "Creating virtual environment (.venv)"
command -v python3 >/dev/null || { echo "Python 3.10+ is required"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python

step "Installing dependencies"
$PY -m pip install --upgrade pip >/dev/null
$PY -m pip install -r requirements-dev.txt
if [ "$GPU" = 1 ]; then $PY -m pip install -r requirements-gpu.txt || echo "mamba-ssm build failed - continuing with the pure-PyTorch Mamba-2"; fi

[ -f .env ] || { cp .env.example .env; echo "Created .env (GEMINI_API_KEY / ECG_PACKAGE_PATH are optional)."; }

if [ "$SKIP_TRAIN" = 0 ] && [ ! -f checkpoints/astroguard_v1.pt ]; then
  step "Training AstroGuard-X on simulated crews"
  $PY -m models.train --samples 3000 --epochs 10
fi
if [ "$SKIP_TESTS" = 0 ]; then step "Running unit tests"; $PY -m pytest -q; fi

if [ "$SKIP_GIT" = 0 ]; then
  step "Initialising Git repository"
  if command -v git >/dev/null; then
    [ -d .git ] || git init -b main >/dev/null
    git add -A
    git diff --cached --quiet || git commit -m "Initial commit: AstroGuard-X" >/dev/null
    echo "Git repository ready. Publish with: bash scripts/publish_github.sh AstroGuard-X"
  else
    $PY scripts/git_init_fallback.py
  fi
fi

step "Done"
echo "Dashboard : .venv/bin/python run.py ui"
echo "REST API  : .venv/bin/python run.py api"
