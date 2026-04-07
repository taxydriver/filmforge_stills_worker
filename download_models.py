"""
Download FLUX2 model files from HuggingFace if not already present.

Paths are resolved from COMFY_MODELS_DIR env var (default: /workspace/models).
Skipped automatically for files that already exist.

Model URLs sourced from gpu_worker/asset_registry.py (same files, proven to work).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_MODELS_DIR = os.getenv("COMFY_MODELS_DIR", "/workspace/models")

MODELS = [
    {
        "name": "flux2-vae",
        "path": f"{_MODELS_DIR}/vae/flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
    },
    {
        "name": "flux2_dev_fp8mixed (UNet)",
        "path": f"{_MODELS_DIR}/unet/flux2_dev_fp8mixed.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/diffusion_models/flux2_dev_fp8mixed.safetensors",
    },
    {
        "name": "mistral_3_small_flux2_bf16 (CLIP)",
        "path": f"{_MODELS_DIR}/clip/mistral_3_small_flux2_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors",
    },
]


def _download(name: str, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download_models] Downloading {name}…", flush=True)
    hf_token = os.getenv("HF_TOKEN", "")
    cmd = [
        "wget",
        "-c",                    # resume if partial file exists
        "--progress=dot:giga",   # log-friendly: one line per 1GB
        "--tries=5",
        "--waitretry=10",
        "--timeout=120",
        "-O", str(dest),
    ]
    if hf_token:
        cmd += [f"--header=Authorization: Bearer {hf_token}"]
    cmd.append(url)
    result = subprocess.run(cmd)
    if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name}")
    size_gb = dest.stat().st_size / 1e9
    print(f"[download_models] Done: {dest.name} ({size_gb:.2f} GB)", flush=True)


def main() -> None:
    missing = [m for m in MODELS if not Path(m["path"]).exists()]
    if not missing:
        print("[download_models] All models present — skipping download.", flush=True)
        return

    print(f"[download_models] {len(missing)} model(s) to download…", flush=True)
    for m in missing:
        _download(m["name"], m["url"], Path(m["path"]))

    print("[download_models] All models ready.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[download_models] FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
