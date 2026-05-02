"""Tool definitions and routing (MCP-style) for the RemoteAgent LLM."""

from __future__ import annotations

from typing import Dict, List

MCP_TOOLS: Dict[str, List[str]] = {
    "skysense_detection": ["image_path", "classes"],
    "referring_expression_segmentation": ["image_path", "prompt"],
    "semantic_segmentation": ["image_path", "classes"],
    "change3d_bcd": ["pre_image_path", "post_image_path"],
    "change3d_scd": ["pre_image_path", "post_image_path"],
    "change3d_bda": ["pre_image_path", "post_image_path"],
    "sm3det_oriented_detection": ["image_path", "classes"],
    "crossearth_semantic_segmentation": ["image_path", "classes"],
    "contour_extraction": ["image_path"],
    "region_contour_extraction": ["image_path", "box"],
    "subobject_contour_extraction": ["image_path"],
    "region_subobject_contour_extraction": ["image_path", "box"],
}

TOOL_TO_SERVICE: Dict[str, str] = {
    "skysense_detection": "skysense_det",
    "referring_expression_segmentation": "remotesam",
    "semantic_segmentation": "remotesam",
    "change3d_bcd": "change3d",
    "change3d_scd": "change3d",
    "change3d_bda": "change3d",
    "sm3det_oriented_detection": "sm3det",
    "crossearth_semantic_segmentation": "crossearth",
    "contour_extraction": "directsam",
    "region_contour_extraction": "directsam",
    "subobject_contour_extraction": "directsam",
    "region_subobject_contour_extraction": "directsam",
}

# Prompt function names (prompt.txt) -> internal canonical keys in MCP_TOOLS / executor.
PROMPT_TOOL_ALIAS: Dict[str, str] = {
    "object_detection": "skysense_detection",
    "binary_change_detection": "change3d_bcd",
    "semantic_change_detection": "change3d_scd",
    "building_damage_assessment": "change3d_bda",
    "oriented_object_detection": "sm3det_oriented_detection",
}

CANONICAL_TO_PROMPT: Dict[str, str] = {v: k for k, v in PROMPT_TOOL_ALIAS.items()}
