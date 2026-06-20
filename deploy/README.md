# DRAW hospital deployment

Always-on, crash-recoverable GPU pipeline in a container, auto-starting on boot. Works
on **Linux (systemd)** and **Windows (Scheduled Task)**. Everything lives in one folder.

## The folder layout

Pick a deploy folder, e.g. `~/work/draw`, containing:

```
~/work/draw/
├── .env                  # you edit this — all config, incl. WATCH_DIR
├── docker-compose.yml    # copied from the repo
├── install-linux.sh      # or install-windows.ps1
├── draw-pipeline.service # systemd unit (Linux)
├── data/                 # model weights + SQLite DB   (created, mounted)
├── output/               # RT-Struct outputs            (created, mounted)
└── logs/                 # rotated+gzipped logs, 30-day (created, mounted)
```

`WATCH_DIR` (set in `.env`) is a **separate** folder the scanner/PACS writes DICOM into —
typically outside the deploy folder (a network share or ingest dir). The container mounts
it read-only and watches it.

## Setup (one folder, three steps)

1. **Copy the deploy files into your folder:**
   ```bash
   mkdir -p ~/work/draw && cd ~/work/draw
   cp /path/to/repo/docker-compose.yml .
   cp /path/to/repo/deploy/* .
   ```

2. **Configure** — copy the template and edit `WATCH_DIR` (and the image tag):
   ```bash
   cp .env.example .env
   nano .env            # set WATCH_DIR=/your/ingest/folder, DRAW_IMAGE=...
   ```

3. **Install the always-on service:**

   **Linux:**
   ```bash
   ./install-linux.sh
   ```
   **Windows** (admin PowerShell, Docker Desktop running):
   ```powershell
   .\install-windows.ps1
   ```

That's it. The pipeline is running, will auto-restart on crash, and starts on boot.

## What you get

- **Always on:** the container uses `restart: unless-stopped`; Linux adds a systemd unit
  (`Restart=always`), Windows a startup Scheduled Task. A reboot or crash brings it back.
- **Crash-recoverable:** a study left mid-prediction by a killed container is re-queued by
  the lease/reaper on restart, and the RT-Struct write is idempotent — so recovery never
  duplicates output.
- **Logs:** 30-day retention, daily rotation, gzip-compressed (only today's is plain),
  written to `logs/draw.log` on the host. Docker's own stdout log is capped separately.
- **No PHI in images:** weights, DICOM, DB, and outputs are all host volumes.

## Operating it

| | Linux | Windows |
|---|---|---|
| App logs | `tail -f logs/draw.log` | `Get-Content -Wait logs\draw.log` |
| Service status | `systemctl status draw-pipeline` | `docker compose ps` |
| Restart | `sudo systemctl restart draw-pipeline` | `docker compose restart` |
| Stop | `sudo systemctl stop draw-pipeline` | `docker compose down` |
| Update image | edit `DRAW_IMAGE`/tag in `.env`, then restart | same |

## Prerequisites

- An NVIDIA GPU with a driver new enough for CUDA 11.8 (≥ 520).
- **Linux:** Docker + Compose v2 + the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- **Windows:** Docker Desktop (WSL2 backend) with GPU support enabled.
