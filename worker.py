"""
Vast.ai PyWorker entry point for FilmForge FLUX2 stills.

Vast.ai clones PYWORKER_REPO and runs: python worker.py
This file therefore:
  1. Downloads FLUX2 models if needed
  2. Starts vastai_server.py (FastAPI backend) as a background subprocess
  3. Configures PyWorker to forward /generate/sync → localhost:8000

Architecture:
  Client → /route/ → worker_url:5000/generate/sync  (PyWorker, public)
                   → http://127.0.0.1:8000/generate/sync  (vastai_server.py, internal)
                   → ComfyUI :8188 (internal)
"""
from __future__ import annotations

import subprocess
import sys
import os
import shutil
import time

# ── 0. Swap in Vast.ai manifest (different model paths from RunPod) ────────────
_here = os.path.dirname(os.path.abspath(__file__))
_vastai_manifest = os.path.join(_here, "manifest.vastai.yaml")
_manifest = os.path.join(_here, "manifest.yaml")
if os.path.exists(_vastai_manifest):
    shutil.copy(_vastai_manifest, _manifest)
    print("[worker] Using manifest.vastai.yaml", flush=True)

# ── 1. Download models if not already on network volume ───────────────────────
print("[worker] Checking FLUX2 models…", flush=True)
subprocess.run([sys.executable, "download_models.py"], check=True)

# ── 2. Start FastAPI backend (vastai_server.py) in background ─────────────────
print("[worker] Starting FilmForge stills server on port 8000…", flush=True)
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "vastai_server:app",
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"],
    env={**os.environ, "PYTHONPATH": os.getcwd()},
)

# Brief pause to let uvicorn bind before PyWorker starts routing
time.sleep(2)

# ── 3. PyWorker (public-facing, port 5000) ────────────────────────────────────
from vastai import Worker, WorkerConfig, HandlerConfig  # noqa: E402

worker_config = WorkerConfig(
    url="http://127.0.0.1:8000",   # our FastAPI backend
    port=5000,                      # PyWorker's public-facing port
    routes=[
        HandlerConfig(
            route="/generate/sync",
            request_parser=lambda data: data,   # forward payload as-is to FastAPI
            workload=lambda data: 100.0,        # fixed cost per still request
            allow_parallel_requests=False,      # ComfyUI is single-threaded
            max_queue_time_secs=600,
        )
    ],
)

Worker(worker_config).run()
