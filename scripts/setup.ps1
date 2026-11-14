<#
.SYNOPSIS
  One-shot setup for AstroGuard-X on Windows (PowerShell 5.1+ / 7+).

.DESCRIPTION
  1. creates a virtual environment (.venv) and installs requirements.txt
  2. creates .env from .env.example (never overwrites an existing one)
  3. trains the compact AstroGuard-X network on simulated crews (skip with -SkipTrain)
  4. runs the unit tests (skip with -SkipTests)
  5. initialises a Git repository with a first commit (skip with -SkipGit)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -SkipTrain -SkipTests
#>
param(
    [switch]$SkipTrain,
    [switch]$SkipTests,
    [switch]$SkipGit,
    [switch]$UseSystemSitePackages   # reuse already-installed torch/streamlit instead of downloading them again
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ---- 1. virtual environment ------------------------------------------------------------
Step "Creating virtual environment (.venv)"
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "Python 3.10+ was not found on PATH. Install it from https://www.python.org/downloads/" }
$ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$ver -lt [version]"3.10") { throw "Python 3.10+ required (found $ver)" }
if (-not (Test-Path ".venv")) {
    if ($UseSystemSitePackages) { & python -m venv --system-site-packages .venv } else { & python -m venv .venv }
}
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"

Step "Installing dependencies"
& $venvPy -m pip install --upgrade pip | Out-Null
& $venvPy -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ---- 2. configuration ------------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env (add GEMINI_API_KEY and ECG_PACKAGE_PATH there - both are optional)."
}

# ---- 3. model --------------------------------------------------------------------------
if (-not $SkipTrain -and -not (Test-Path "checkpoints\astroguard_v2.pt")) {
    Step "Training AstroGuard-X on simulated crews (CPU: a few minutes)"
    & $venvPy -m models.train --samples 12000 --epochs 25
    if ($LASTEXITCODE -ne 0) { throw "training failed" }
}

# ---- 4. tests --------------------------------------------------------------------------
if (-not $SkipTests) {
    Step "Running unit tests"
    & $venvPy -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw "tests failed" }
}

# ---- 5. git ------------------------------------------------------------------------------
if (-not $SkipGit) {
    Step "Initialising Git repository"
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) {
        if (-not (Test-Path ".git")) { git init -b main | Out-Null }
        git add -A
        git diff --cached --quiet
        if ($LASTEXITCODE -ne 0) { git commit -m "Initial commit: AstroGuard-X" | Out-Null }
        Write-Host "Git repository ready. Publish with: scripts\publish_github.ps1 -Repo AstroGuard-X"
    } else {
        Write-Host "git.exe not found - using the pure-Python fallback (dulwich) for the first commit."
        & $venvPy scripts\git_init_fallback.py
        Write-Host "Tip: install Git (winget install --id Git.Git) to push to GitHub."
    }
}

Step "Done"
Write-Host "Start the dashboard : .\.venv\Scripts\python.exe run.py ui"
Write-Host "Start the REST API  : .\.venv\Scripts\python.exe run.py api"
