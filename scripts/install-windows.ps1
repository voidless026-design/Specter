# Installs the built ev.exe to a per-user location and registers it to
# start automatically at logon via Task Scheduler.
#
# Run scripts\build-windows.ps1 first (on this same Windows machine - the
# build step cannot be done anywhere else). Then from PowerShell at the
# repo root:
#   .\scripts\install-windows.ps1
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    Write-Error "This must run on native Windows."
    exit 1
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuiltExe = Join-Path $RepoRoot "dist\ev\ev.exe"
if (-not (Test-Path $BuiltExe)) {
    Write-Error "dist\ev\ev.exe not found - run .\scripts\build-windows.ps1 first."
    exit 1
}

$InstallDir = Join-Path $env:LOCALAPPDATA "EV-Assistant"
# Matches config.py's platformdirs call (appauthor=False), so this is
# where `ev.exe` itself will look for config.toml once running.
$ConfigDir = Join-Path $env:LOCALAPPDATA "ev-assistant"

Write-Host "==> Installing to $InstallDir"
if (Test-Path $InstallDir) {
    Remove-Item -Recurse -Force $InstallDir
}
Copy-Item -Recurse (Join-Path $RepoRoot "dist\ev") $InstallDir

$EnvFile = Join-Path $InstallDir "ev.env"
if (-not (Test-Path $EnvFile)) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
    @"
ANTHROPIC_API_KEY=sk-ant-your-key-here
EV_CONTROL_TOKEN=$token
"@ | Set-Content -Path $EnvFile -Encoding utf8
    Write-Host "    Wrote $EnvFile with a generated EV_CONTROL_TOKEN."
    Write-Host "    Edit it now and set your real ANTHROPIC_API_KEY."
} else {
    Write-Host "    $EnvFile already exists - leaving it as-is."
}

# Task Scheduler has no native "load an env file" support, so this small
# wrapper loads ev.env into the process environment before launching ev.exe.
$LauncherPath = Join-Path $InstallDir "run-ev-daemon.ps1"
@'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-Content (Join-Path $here "ev.env") | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}
& (Join-Path $here "ev.exe") daemon
'@ | Set-Content -Path $LauncherPath -Encoding utf8

Write-Host "==> Registering scheduled task 'EV-Assistant' (runs at logon)"
Unregister-ScheduledTask -TaskName "EV-Assistant" -Confirm:$false -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$LauncherPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "EV-Assistant" -Action $action -Trigger $trigger -Settings $settings `
    -Description 'E.V. voice assistant - listens for "Hey E.V." in the background' | Out-Null

Write-Host ""
Write-Host "Installed. Edit $EnvFile and set ANTHROPIC_API_KEY, then either:"
Write-Host "  - log off and back on (or restart), or"
Write-Host "  - run now:  Start-ScheduledTask -TaskName EV-Assistant"
Write-Host ""
Write-Host 'Say "Hey E.V." to talk to her, or from any shell (including SSH into this PC):'
Write-Host "  $InstallDir\ev.exe ask `"what's the weather like`""
Write-Host ""
Write-Host "To change personality, wake phrases, or news feeds, edit the config file at:"
Write-Host "  $ConfigDir\config.toml   (created on first run)"
