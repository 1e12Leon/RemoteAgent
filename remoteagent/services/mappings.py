"""Map tool names to backend API task identifiers."""

from __future__ import annotations

from typing import Dict

REMOTE_SAM_TOOL_TO_TASK: Dict[str, str] = {
    "referring_expression_segmentation": "referring_seg",
    "semantic_segmentation": "semantic_seg",
}

CHANGE3D_TOOL_TO_TASK: Dict[str, str] = {
    "change3d_bcd": "bcd",
    "change3d_scd": "scd",
    "change3d_bda": "bda",
}

SM3DET_TOOL_TO_TASK: Dict[str, str] = {
    "sm3det_oriented_detection": "oriented_detection",
}

CROSSEARTH_TOOL_TO_TASK: Dict[str, str] = {
    "crossearth_semantic_segmentation": "semantic_seg",
}

SKYSENSE_DET_TASK = "detection"
