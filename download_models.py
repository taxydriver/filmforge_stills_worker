"""
Download FLUX2 model files from HuggingFace if not already present.

Paths are resolved from COMFY_MODELS_DIR env var (default: /workspace/models).
Skipped automatically for files that already exist.

Model URLs sourced from gpu_worker/asset_registry.py (same files, proven to work).
"""
from __future__ import annotations

import os
import sys
import urllib.request
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
    tmp = dest.with_suffix(".part")
    print(f"[download_models] Downloading {name}…", flush=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            gb = downloaded / 1e9
            total_gb = total_size / 1e9
            print(f"\r  {pct:.1f}%  {gb:.2f}/{total_gb:.2f} GB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name}: {exc}") from exc

    print()  # newline after progress
    tmp.rename(dest)
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
