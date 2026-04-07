#!/bin/bash
set -e

# Download FLUX2 models BEFORE ComfyUI starts so it can see them on boot.
# On subsequent Flash Boot restores, models already exist — this is a fast no-op.
echo "[filmforge] Downloading models before ComfyUI starts..."
python /app/download_models.py

echo "[filmforge] Models ready. Handing off to RunPod start.sh..."
exec /start.sh
