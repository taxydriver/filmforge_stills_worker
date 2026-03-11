from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "manifest.yaml"


@lru_cache(maxsize=1)
def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Manifest root must be a mapping")
    return data


def supported_workflows() -> List[str]:
    manifest = load_manifest()
    workflows = manifest.get("workflows") or {}
    if not isinstance(workflows, dict):
        return []
    return sorted(workflows.keys())


def workflow_exists(workflow_name: str) -> bool:
    return workflow_name in supported_workflows()


def get_workflow_entry(workflow_name: str) -> Dict[str, Any]:
    manifest = load_manifest()
    workflows = manifest.get("workflows") or {}
    entry = workflows.get(workflow_name)
    if not isinstance(entry, dict):
        raise KeyError(f"Unsupported workflow_name={workflow_name!r}")
    return entry


def get_workflow_path(workflow_name: str) -> Path:
    entry = get_workflow_entry(workflow_name)
    rel = entry.get("file")
    if not isinstance(rel, str) or not rel.strip():
        raise RuntimeError(f"Workflow {workflow_name!r} is missing 'file' in manifest")
    path = BASE_DIR / rel
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found for {workflow_name!r}: {path}")
    return path


def load_workflow_template(workflow_name: str) -> Dict[str, Any]:
    path = get_workflow_path(workflow_name)
    with path.open("r", encoding="utf-8") as f:
        graph = json.load(f)
    if not isinstance(graph, dict):
        raise RuntimeError(f"Workflow {path} must be a node dict")
    return graph


def get_required_model_files() -> List[str]:
    manifest = load_manifest()
    models = manifest.get("models") or {}
    required = models.get("required_files") if isinstance(models, dict) else None
    if not isinstance(required, list):
        return []
    return [str(x) for x in required if isinstance(x, str) and x.strip()]


def get_required_nodes(workflow_name: str) -> List[str]:
    entry = get_workflow_entry(workflow_name)
    req_nodes = entry.get("required_nodes")
    if isinstance(req_nodes, list) and req_nodes:
        return [str(x) for x in req_nodes if isinstance(x, str) and x.strip()]

    graph = load_workflow_template(workflow_name)
    found: set[str] = set()
    for node in graph.values():
        if isinstance(node, dict):
            class_type = node.get("class_type")
            if isinstance(class_type, str) and class_type:
                found.add(class_type)
    return sorted(found)


def workflow_filename(workflow_name: str) -> str:
    return get_workflow_path(workflow_name).name
