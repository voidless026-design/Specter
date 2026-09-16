# Builds ev.exe on Windows. Run from PowerShell at the repo root:
#   .\scripts\build-windows.ps1
#
# PyInstaller cannot cross-compile - this MUST run on an actual Windows
# machine (not WSL, not a Linux box). Requires Python 3.11+ for Windows
# already installed (python.org or the Microsoft Store).
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This must run on native Windows - PyInstaller does not cross-compile."
    exit 1
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$pythonCmd = "py"
$pythonArgs = @("-3")
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    $pythonCmd = "python"
    $pythonArgs = @()
}

if (-not (Test-Path ".\.venv-build")) {
    Write-Host "==> Creating build virtualenv..."
    & $pythonCmd @pythonArgs -m venv .venv-build
}

Write-Host "==> Installing E.V. and PyInstaller into the build venv..."
& .\.venv-build\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv-build\Scripts\python.exe -m pip install -e . pyinstaller | Out-Null

Write-Host "==> Running PyInstaller..."
& .\.venv-build\Scripts\pyinstaller.exe scripts\ev.spec --noconfirm --distpath dist --workpath build

Write-Host ""
Write-Host "Built: $RepoRoot\dist\ev\ev.exe"
Write-Host "Next: .\scripts\install-windows.ps1 to register autostart, or just run ev.exe directly."
