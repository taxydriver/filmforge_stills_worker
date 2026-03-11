from __future__ import annotations

from typing import Any, Dict


def extract_first_image_artifact(history_entry: Dict[str, Any]) -> Dict[str, str]:
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("History entry missing outputs")

    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        images = node_output.get("images")
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            filename = image.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            return {
                "filename": filename,
                "subfolder": str(image.get("subfolder") or ""),
                "type": str(image.get("type") or "output"),
                "source_node_id": str(node_id),
            }

    raise RuntimeError("No image outputs found in history entry")
