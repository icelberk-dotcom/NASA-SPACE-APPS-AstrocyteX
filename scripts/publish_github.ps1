<#
.SYNOPSIS
  Publish AstroGuard-X to GitHub (public by default).

.DESCRIPTION
  * With the GitHub CLI (`gh`, already logged in):   creates the repo and pushes.
  * Otherwise: pass -RemoteUrl of an EMPTY repository you created at https://github.com/new and the
    script adds it as `origin` and pushes `main`.

.EXAMPLE
  scripts\publish_github.ps1 -Repo NASA-space-apps-Ondankoymacano
  scripts\publish_github.ps1 -RemoteUrl https://github.com/icelberk-dotcom/NASA-space-apps-Ondankoymacano.git
  scripts\publish_github.ps1 -Repo NASA-space-apps-Ondankoymacano -Private
#>
param(
    [string]$Repo = "NASA-space-apps-Ondankoymacano",
    [string]$RemoteUrl = "",
    [switch]$Private
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    foreach ($candidate in @("C:\Program Files\Git\cmd", "$env:LOCALAPPDATA\Programs\Git\cmd")) {
        if (Test-Path (Join-Path $candidate "git.exe")) {
            $env:PATH = "$candidate;$env:PATH"
            break
        }
    }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed. Install it (winget install --id Git.Git), reopen the terminal and re-run."
}
if (-not (Test-Path ".git")) { git init -b main | Out-Null }
if (-not (git config user.name 2>$null)) { git config user.name "icelberk-dotcom" }
if (-not (git config user.email 2>$null)) { git config user.email "icelberk-dotcom@users.noreply.github.com" }

git add -A

# Safety: make sure no secrets are staged or committed before creating a commit or pushing.
$tracked = git ls-files
if ($tracked -contains ".env") { throw ".env is tracked by git - remove it (git rm --cached .env) before publishing." }
if ((git grep --cached -I -n -E "(AIza[0-9A-Za-z_\-]{30,}|AQ\.[0-9A-Za-z_\-]{20,})" 2>$null)) { throw "Possible Google API key found in staged/tracked files - aborting." }

git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m "Update AstroGuard-X" | Out-Null }

if ($RemoteUrl) {
    if (git remote | Select-String -SimpleMatch "origin") { git remote set-url origin $RemoteUrl } else { git remote add origin $RemoteUrl }
    git push -u origin main
}
elseif (Get-Command gh -ErrorAction SilentlyContinue) {
    $vis = if ($Private) { "--private" } else { "--public" }
    gh repo create $Repo $vis --source . --remote origin --push `
        --description "AI-powered astronaut health monitoring with gamified mitigation tasks (NASA Space Apps Challenge)"
}
else {
    Write-Host @"
GitHub CLI not found. Do this once:
  1. Create an EMPTY repo at https://github.com/new  (name: $Repo, visibility: Public)
  2. Run:  scripts\publish_github.ps1 -RemoteUrl https://github.com/<your-user>/$Repo.git
"@
    exit 1
}
Write-Host "Published. Add topics (nasa-space-apps, astronaut-health, pytorch, mamba, streamlit) in the repo settings."
