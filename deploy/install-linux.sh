#!/usr/bin/env bash
#
# One-shot DRAW deployment for a Linux hospital box.
#
# Layout (everything in one folder, e.g. ~/work/draw):
#   .env                 <- you edit this (copied from .env.example)
#   docker-compose.yml   <- shipped
#   data/  output/  logs/ <- created here, mounted into the container
#   WATCH_DIR (in .env)  <- some OTHER folder the scanner writes to
#
# Run from inside the deploy folder:
#   cp .env.example .env && nano .env     # set WATCH_DIR etc.
#   ./install-linux.sh
#
# Installs an always-on, crash-recoverable systemd service that pulls the GPU image
# and runs `docker compose up`. Re-running refreshes config and restarts.
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DEPLOY_DIR"

log() { echo -e "\033[1;34m[draw]\033[0m $*"; }
die() { echo -e "\033[1;31m[draw] ERROR:\033[0m $*" >&2; exit 1; }

# ---- preflight ----------------------------------------------------------------
command -v docker >/dev/null || die "docker not found. Install Docker + the nvidia-container-toolkit."
docker compose version >/dev/null 2>&1 || die "'docker compose' not available (need Compose v2)."
[[ -f docker-compose.yml ]] || die "docker-compose.yml not found next to this script."
[[ -f .env ]] || die "No .env found. Run: cp .env.example .env  then edit WATCH_DIR."

# shellcheck disable=SC1091
set -a; source .env; set +a
[[ -n "${WATCH_DIR:-}" && "$WATCH_DIR" != "/path/to/incoming" ]] || die "Set WATCH_DIR in .env to the scanner's ingest folder."
[[ -d "$WATCH_DIR" ]] || die "WATCH_DIR '$WATCH_DIR' does not exist. Create it or point at the real ingest folder."

# GPU sanity (warn, don't block — driver issues surface clearly in logs otherwise).
if ! docker run --rm --gpus all "${DRAW_IMAGE:-nvidia/cuda:11.8.0-base-ubuntu22.04}" nvidia-smi >/dev/null 2>&1; then
    log "WARNING: could not run nvidia-smi in a container. Check the NVIDIA driver + nvidia-container-toolkit before relying on this."
fi

# ---- folders ------------------------------------------------------------------
mkdir -p "${DATA_DIR:-./data}" "${OUTPUT_DIR:-./output}" "${LOG_DIR:-./logs}"
log "Folders ready: data=${DATA_DIR:-./data} output=${OUTPUT_DIR:-./output} logs=${LOG_DIR:-./logs}"
log "Watching: $WATCH_DIR"

# ---- pull image ---------------------------------------------------------------
log "Pulling image ${DRAW_IMAGE:-draw:gpu} (skip if building locally)…"
docker compose pull || log "Pull skipped/failed — will build locally on first 'up' if needed."

# ---- systemd unit -------------------------------------------------------------
UNIT=/etc/systemd/system/draw-pipeline.service
log "Installing systemd service -> $UNIT (sudo)"
sed "s|@@DEPLOY_DIR@@|$DEPLOY_DIR|g" draw-pipeline.service > /tmp/draw-pipeline.service
sudo cp /tmp/draw-pipeline.service "$UNIT" && rm -f /tmp/draw-pipeline.service
sudo systemctl daemon-reload
sudo systemctl enable draw-pipeline.service
sudo systemctl restart draw-pipeline.service

log "Done — DRAW is running and will auto-start on boot."
log "Logs (app):    tail -f ${LOG_DIR:-./logs}/draw.log"
log "Logs (service):journalctl -u draw-pipeline -f"
log "Status:        systemctl status draw-pipeline"
log "Stop:          sudo systemctl stop draw-pipeline"
