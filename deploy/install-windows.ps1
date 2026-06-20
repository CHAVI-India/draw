# One-shot DRAW deployment for a Windows hospital box.
#
# Windows has no systemd. The equivalent always-on setup is:
#   * the container's `restart: unless-stopped` keeps it alive across crashes, and
#   * a Scheduled Task at system startup runs `docker compose up -d` after a reboot.
#
# Prereqs: Docker Desktop (WSL2 backend) with GPU support enabled, and a recent NVIDIA
# driver. Run this from inside the deploy folder in an ADMIN PowerShell:
#   Copy-Item .env.example .env ; notepad .env     # set WATCH_DIR etc.
#   .\install-windows.ps1
#
# Re-running refreshes the task and restarts the stack.
$ErrorActionPreference = "Stop"
$DeployDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $DeployDir

function Info($m) { Write-Host "[draw] $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "[draw] ERROR: $m" -ForegroundColor Red; exit 1 }

# ---- preflight ----------------------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die "docker not found. Install Docker Desktop." }
docker compose version | Out-Null
if (-not (Test-Path "docker-compose.yml")) { Die "docker-compose.yml not found next to this script." }
if (-not (Test-Path ".env")) { Die "No .env found. Run: Copy-Item .env.example .env ; then edit WATCH_DIR." }

# Parse .env into a hashtable (KEY=VALUE lines, ignore comments/blanks).
$envVars = @{}
Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([^#=]+?)\s*=\s*(.*)\s*$') { $envVars[$matches[1]] = $matches[2] }
}
$watch = $envVars["WATCH_DIR"]
if (-not $watch -or $watch -eq "/path/to/incoming") { Die "Set WATCH_DIR in .env to the scanner's ingest folder." }
if (-not (Test-Path $watch)) { Die "WATCH_DIR '$watch' does not exist." }

# ---- folders ------------------------------------------------------------------
foreach ($k in @("DATA_DIR","OUTPUT_DIR","LOG_DIR")) {
    $p = $envVars[$k]; if (-not $p) { $p = ".\$($k.Split('_')[0].ToLower())" }
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}
Info "Watching: $watch"

# ---- pull + start now ---------------------------------------------------------
Info "Pulling image and starting the stack…"
docker compose pull
docker compose up -d

# ---- Scheduled Task: start at boot --------------------------------------------
$taskName = "DRAW-Pipeline"
$dockerCompose = "docker compose up -d"
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c cd /d `"$DeployDir`" && $dockerCompose"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Info "Registering startup task '$taskName' (admin)…"
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Info "Done — DRAW is running and will auto-start on boot."
Info "Logs (app):     Get-Content -Wait $($envVars['LOG_DIR'])\draw.log"
Info "Status:         docker compose ps"
Info "Stop:           docker compose down  (and disable task: Unregister-ScheduledTask -TaskName $taskName)"
