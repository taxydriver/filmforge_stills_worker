from __future__ import annotations

import os
from pathlib import Path
from typing import List

from runtime import ComfyRuntime
from workflow_registry import (
    get_required_model_files,
    get_required_nodes,
    get_workflow_path,
    supported_workflows,
)


def _models_dir() -> Path:
    return Path(os.getenv("COMFY_MODELS_DIR") or "/workspace/ComfyUI/models")


def run_startup_preflight(runtime: ComfyRuntime) -> None:
    errors: List[str] = []

    workflow_names = supported_workflows()
    if not workflow_names:
        errors.append("No workflows found in manifest")
    else:
        print(f"[preflight] workflows declared: {workflow_names}")

    for workflow_name in workflow_names:
        try:
            path = get_workflow_path(workflow_name)
            print(f"[preflight] workflow ok: {workflow_name} -> {path}")
        except Exception as exc:
            errors.append(f"Workflow {workflow_name!r} invalid: {exc}")

    models_root = _models_dir()
    for rel in get_required_model_files():
        model_path = models_root / rel
        if not model_path.exists():
            errors.append(f"Missing model file: {model_path}")
        else:
            print(f"[preflight] model ok: {model_path}")

    try:
        runtime.wait_until_ready(timeout_seconds=180)
        object_info = runtime.get_object_info()
        available_nodes = set(object_info.keys())
    except Exception as exc:  # pragma: no cover - runtime dependent
        errors.append(f"Comfy runtime unavailable: {exc}")
        available_nodes = set()

    for workflow_name in workflow_names:
        required_nodes = get_required_nodes(workflow_name)
        missing_nodes = sorted(node for node in required_nodes if node not in available_nodes)
        if missing_nodes:
            errors.append(
                "Missing nodes for workflow "
                f"{workflow_name!r}: {', '.join(missing_nodes)}"
            )
        else:
            print(
                f"[preflight] nodes ok: {workflow_name} "
                f"({len(required_nodes)} required)"
            )

    if errors:
        message = "Startup preflight failed:\n- " + "\n- ".join(errors)
        raise RuntimeError(message)

    print("[preflight] startup checks passed")
