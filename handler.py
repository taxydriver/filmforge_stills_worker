from __future__ import annotations

import base64
import json
import os
import traceback
from typing import Any, Dict

import runpod
from pydantic import ValidationError

from flux2_ref import patch_flux2_ref_workflow
from flux2_text import patch_flux2_text_workflow
from output_parser import extract_first_image_artifact
from preflight import run_startup_preflight
from still_requests import StillRequest
from runtime import ComfyRuntime
from workflow_registry import load_workflow_template, workflow_filename

RUNTIME = ComfyRuntime()


def _startup_preflight() -> None:
    print("[startup] running preflight", flush=True)
    run_startup_preflight(RUNTIME)


_startup_preflight()


def _run_flux2_text(request: StillRequest) -> Dict[str, Any]:
    workflow_template = load_workflow_template(request.workflow_name)
    workflow_graph, patched_nodes = patch_flux2_text_workflow(workflow_template, request)

    prompt_id = RUNTIME.submit_workflow(workflow_graph)
    history_entry = RUNTIME.poll_history_entry(prompt_id)
    image_artifact = extract_first_image_artifact(history_entry)

    image_bytes = RUNTIME.fetch_image_bytes(
        image_artifact["filename"],
        subfolder=image_artifact.get("subfolder", ""),
        image_type=image_artifact.get("type", "output"),
    )

    return {
        "status": "ok",
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "filename": image_artifact["filename"],
        "workflow_name": request.workflow_name,
        "workflow_file": workflow_filename(request.workflow_name),
        "patched_nodes": patched_nodes,
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


def _run_flux2_ref(request: StillRequest) -> Dict[str, Any]:
    workflow_template = load_workflow_template(request.workflow_name)

    ref_payload = request.ref_images[0]
    ref_bytes = _decode_ref_image_payload(ref_payload)
    uploaded_name = RUNTIME.upload_input_image(image_bytes=ref_bytes, filename="ref.png")
    print(f"[flux2_ref] uploaded ref image name={uploaded_name}")

    workflow_graph, patched_nodes = patch_flux2_ref_workflow(
        workflow_template,
        request,
        ref_image_name=uploaded_name,
    )

    prompt_id = RUNTIME.submit_workflow(workflow_graph)
    history_entry = RUNTIME.poll_history_entry(prompt_id)
    image_artifact = extract_first_image_artifact(history_entry)

    image_bytes = RUNTIME.fetch_image_bytes(
        image_artifact["filename"],
        subfolder=image_artifact.get("subfolder", ""),
        image_type=image_artifact.get("type", "output"),
    )

    return {
        "status": "ok",
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "filename": image_artifact["filename"],
        "workflow_name": request.workflow_name,
        "workflow_file": workflow_filename(request.workflow_name),
        "patched_nodes": patched_nodes,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = (job or {}).get("input") or {}
        request = StillRequest.model_validate(payload)

        if request.workflow_name == "flux2_text":
            return _run_flux2_text(request)
        if request.workflow_name == "flux2_ref":
            return _run_flux2_ref(request)
        return {
            "status": "error",
            "error": f"Unsupported workflow_name={request.workflow_name!r}",
        }

    except ValidationError as exc:
        return {
            "status": "error",
            "error": "validation_error",
            "details": exc.errors(),
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(exc),
        }


def build_demo_forest_request() -> Dict[str, Any]:
    return {
        "workflow_name": "flux2_text",
        "prompt": (
            "establishing wide shot of a dense ancient forest at dawn, "
            "massive tree trunks, narrow dirt path, wet roots and ferns, "
            "pale sunlight through mist"
        ),
        "negative_prompt": "",
        "width": 1024,
        "height": 576,
        "seed": 42,
        "steps": 28,
        "cfg": 3.5,
        "sampler_name": "euler",
        "scheduler": "simple",
        "filename_prefix": "bench_forest_environment_c1",
    }


if __name__ == "__main__":
    if os.getenv("DEMO_MODE", "0") in {"1", "true", "yes", "on"}:
        print("[demo] running one forest benchmark request")
        response = handler({"input": build_demo_forest_request()})
        # Avoid dumping massive base64 in logs
        image_b64 = response.pop("image_base64", None)
        if isinstance(image_b64, str):
            response["image_base64_len"] = len(image_b64)
        print(json.dumps(response, indent=2))
    else:
        runpod.serverless.start({"handler": handler})
