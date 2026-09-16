# Unregisters the E.V. scheduled task and optionally removes the installed
# files. Run from PowerShell:
#   .\scripts\uninstall-windows.ps1
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This must run on native Windows."
    exit 1
}

Unregister-ScheduledTask -TaskName "EV-Assistant" -Confirm:$false -ErrorAction SilentlyContinue
Write-Host "Removed scheduled task."

$InstallDir = Join-Path $env:LOCALAPPDATA "EV-Assistant"
# On Windows, config.toml and E.V.'s memory/speech model share one
# platformdirs path (no separate config-vs-data split like on Linux).
$ConfigDir = Join-Path $env:LOCALAPPDATA "ev-assistant"

$reply = Read-Host "Delete $InstallDir (exe, API key) and $ConfigDir (config, memory, speech model)? [y/N]"
if ($reply -match '^[Yy]') {
    Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $ConfigDir -ErrorAction SilentlyContinue
    Write-Host "Removed everything."
} else {
    Write-Host "Left $InstallDir and $ConfigDir in place."
}
