"""
Vast.ai-compatible HTTP server for FilmForge FLUX2 stills.

Exposes POST /generate/sync — Vast.ai routes requests here after /route/.
Same internal logic as handler.py (RunPod version) — only the transport changes.

Request body:  StillRequest JSON (workflow_name, prompt, width, height, seed, ...)
Response body: {"images": [{"filename": "...", "data": "<base64 PNG>"}]}
"""
from __future__ import annotations

import asyncio
import base64
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from flux2_ref import patch_flux2_ref_workflow
from flux2_text import patch_flux2_text_workflow
from output_parser import extract_first_image_artifact
from preflight import run_startup_preflight
from runtime import ComfyRuntime
from still_requests import StillRequest
from workflow_registry import load_workflow_template, workflow_filename

_runtime: ComfyRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime
    print("[vastai_server] Initialising ComfyRuntime and running preflight…")
    _runtime = ComfyRuntime()
    # run_startup_preflight blocks (waits for ComfyUI to be ready + validates models)
    # asyncio.to_thread keeps the event loop responsive during the wait
    await asyncio.to_thread(run_startup_preflight, _runtime)
    print("[vastai_server] Ready to serve requests.")
    yield


app = FastAPI(title="FilmForge FLUX2 Stills — Vast.ai", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate/sync")
def generate_sync(req: StillRequest) -> JSONResponse:
    """
    Main inference endpoint.
    FastAPI runs sync endpoints in a thread — appropriate here since
    FLUX2 generation blocks for 30-60 s and ComfyUI is single-threaded.
    """
    try:
        result = _dispatch(req)
        return JSONResponse(content=result)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ── internal dispatch ────────────────────────────────────────────────────────

def _dispatch(req: StillRequest) -> dict:
    if req.workflow_name == "flux2_text":
        return _run_flux2_text(req)
    if req.workflow_name == "flux2_ref":
        return _run_flux2_ref(req)
    raise ValueError(f"Unsupported workflow_name={req.workflow_name!r}")


def _run_flux2_text(request: StillRequest) -> dict:
    workflow_template = load_workflow_template(request.workflow_name)
    workflow_graph, patched_nodes = patch_flux2_text_workflow(workflow_template, request)

    prompt_id = _runtime.submit_workflow(workflow_graph)
    history_entry = _runtime.poll_history_entry(prompt_id)
    image_artifact = extract_first_image_artifact(history_entry)

    image_bytes = _runtime.fetch_image_bytes(
        image_artifact["filename"],
        subfolder=image_artifact.get("subfolder", ""),
        image_type=image_artifact.get("type", "output"),
    )

    return {
        "images": [{
            "filename": image_artifact["filename"],
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }],
        "workflow_name": request.workflow_name,
        "workflow_file": workflow_filename(request.workflow_name),
    }


def _decode_ref_image_payload(value: str) -> bytes:
    payload = (value or "").strip()
    if not payload:
        raise RuntimeError("ref_images contains an empty payload")
    if "," in payload and payload.lower().startswith("data:image/"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        raise RuntimeError(f"Invalid ref image base64 payload: {exc}") from exc


def _run_flux2_ref(request: StillRequest) -> dict:
    workflow_template = load_workflow_template(request.workflow_name)

    ref_bytes = _decode_ref_image_payload(request.ref_images[0])
    uploaded_name = _runtime.upload_input_image(image_bytes=ref_bytes, filename="ref.png")
    print(f"[flux2_ref] uploaded ref image name={uploaded_name!r}")

    workflow_graph, patched_nodes = patch_flux2_ref_workflow(
        workflow_template,
        request,
        ref_image_name=uploaded_name,
    )

    prompt_id = _runtime.submit_workflow(workflow_graph)
    history_entry = _runtime.poll_history_entry(prompt_id)
    image_artifact = extract_first_image_artifact(history_entry)

    image_bytes = _runtime.fetch_image_bytes(
        image_artifact["filename"],
        subfolder=image_artifact.get("subfolder", ""),
        image_type=image_artifact.get("type", "output"),
    )

    return {
        "images": [{
            "filename": image_artifact["filename"],
            "data": base64.b64encode(image_bytes).decode("utf-8"),
        }],
        "workflow_name": request.workflow_name,
        "workflow_file": workflow_filename(request.workflow_name),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
