from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

from still_requests import StillRequest


def patch_flux2_text_workflow(
    workflow: Dict[str, Any],
    request: StillRequest,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    graph = deepcopy(workflow)
    patched: Dict[str, list[str]] = {
        "prompt": [],
        "width": [],
        "height": [],
        "filename_prefix": [],
        "seed": [],
        "steps": [],
        "cfg": [],
        "sampler_name": [],
        "scheduler": [],
    }

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = node.get("class_type")

        if class_type == "CLIPTextEncode" and "text" in inputs:
            if request.negative_prompt:
                inputs["text"] = f"{request.prompt}. Avoid: {request.negative_prompt}."
            else:
                inputs["text"] = request.prompt
            patched["prompt"].append(str(node_id))

        if class_type == "SaveImage" and "filename_prefix" in inputs:
            inputs["filename_prefix"] = request.filename_prefix
            patched["filename_prefix"].append(str(node_id))

        if "width" in inputs:
            inputs["width"] = int(request.width)
            patched["width"].append(str(node_id))

        if "height" in inputs:
            inputs["height"] = int(request.height)
            patched["height"].append(str(node_id))

        if "noise_seed" in inputs:
            inputs["noise_seed"] = int(request.seed)
            patched["seed"].append(str(node_id))
        elif "seed" in inputs:
            inputs["seed"] = int(request.seed)
            patched["seed"].append(str(node_id))

        if "steps" in inputs:
            inputs["steps"] = int(request.steps)
            patched["steps"].append(str(node_id))

        if "cfg" in inputs:
            inputs["cfg"] = float(request.cfg)
            patched["cfg"].append(str(node_id))
        elif "guidance" in inputs:
            inputs["guidance"] = float(request.cfg)
            patched["cfg"].append(str(node_id))

        if "sampler_name" in inputs:
            inputs["sampler_name"] = str(request.sampler_name)
            patched["sampler_name"].append(str(node_id))

        if "scheduler" in inputs:
            inputs["scheduler"] = str(request.scheduler)
            patched["scheduler"].append(str(node_id))

    if not patched["prompt"]:
        raise RuntimeError("Flux2 text workflow patch failed: no CLIPTextEncode text node found")
    if not patched["filename_prefix"]:
        raise RuntimeError("Flux2 text workflow patch failed: no SaveImage filename_prefix node found")

    print(
        "[flux2_text.patch] "
        f"prompt_nodes={patched['prompt']} "
        f"width_nodes={patched['width']} "
        f"height_nodes={patched['height']} "
        f"filename_prefix_nodes={patched['filename_prefix']} "
        f"seed_nodes={patched['seed']} "
        f"steps_nodes={patched['steps']} "
        f"cfg_nodes={patched['cfg']} "
        f"sampler_nodes={patched['sampler_name']} "
        f"scheduler_nodes={patched['scheduler']}"
    )

    return graph, patched
