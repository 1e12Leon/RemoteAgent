"""HTTP routing to external vision tool services."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from remoteagent.config.tools_schema import TOOL_TO_SERVICE
from remoteagent.services.mappings import (
    CHANGE3D_TOOL_TO_TASK,
    CROSSEARTH_TOOL_TO_TASK,
    REMOTE_SAM_TOOL_TO_TASK,
    SKYSENSE_DET_TASK,
    SM3DET_TOOL_TO_TASK,
)
from remoteagent.utils.http import http_post_json
from remoteagent.utils.image import encode_image


class ServiceExecutor:
    """Calls RemoteSAM / Change3D / SM3Det / CrossEarth / SkySense / DirectSAM HTTP APIs."""

    def __init__(
        self,
        api_urls: Dict[str, Optional[str]],
        timeout_default: float = 120.0,
        timeout_change3d: float = 180.0,
    ) -> None:
        self._api_urls = api_urls
        self._timeout_default = timeout_default
        self._timeout_change3d = timeout_change3d

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        service = TOOL_TO_SERVICE.get(tool_name)
        if not service:
            return f"Error: Unknown tool {tool_name}"

        url = self._api_urls.get(service)
        if not url:
            return (
                f"Error: Service URL for {service} is not configured. "
                f"Cannot execute {tool_name}."
            )

        if service == "remotesam":
            return self._call_remote_sam(tool_name, args, url)
        if service == "change3d":
            return self._call_change3d(tool_name, args, url)
        if service == "sm3det":
            return self._call_sm3det(tool_name, args, url)
        if service == "crossearth":
            return self._call_crossearth(tool_name, args, url)
        if service == "skysense_det":
            return self._call_skysense_det(args, url)
        if service == "directsam":
            return self._call_directsam(tool_name, args, url)

        return f"Error: Unimplemented execution logic for service {service}"

    def _call_remote_sam(self, tool_name: str, args: Dict[str, Any], api_url: str) -> str:
        task = REMOTE_SAM_TOOL_TO_TASK.get(tool_name)
        if not task:
            return "Error: Unknown RemoteSAM task."

        image_path = args.get("image_path") or ""
        if not image_path:
            return "Error: Missing image_path."

        try:
            payload = {
                "task": task,
                "image": encode_image(image_path),
                "text": args.get("prompt", args.get("text", "")),
                "classes": args.get("classes") or [],
                "box_threshold": 0.3,
                "text_threshold": 0.25,
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_default)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                if "counts" in data:
                    msg += "\nCounts: " + json.dumps(data["counts"], ensure_ascii=False)
                if "detections" in data:
                    total = sum(len(v) for v in (data["detections"] or {}).values())
                    msg += f"\nDetected {total} objects."
                if "caption" in data:
                    msg += "\nCaption: " + str(data["caption"])
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"RemoteSAM call failed: {e}"

    def _call_change3d(self, tool_name: str, args: Dict[str, Any], api_url: str) -> str:
        task = CHANGE3D_TOOL_TO_TASK.get(tool_name, "bcd")
        pre_path = args.get("pre_image_path") or ""
        post_path = args.get("post_image_path") or ""

        if not pre_path or not post_path:
            return "Error: Change3D requires both pre_image_path and post_image_path."

        try:
            payload = {
                "task": task,
                "pre_image": encode_image(pre_path),
                "post_image": encode_image(post_path),
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_change3d)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                extra = {
                    k: v
                    for k, v in data.items()
                    if k not in ("status", "message", "task")
                    and not (isinstance(v, str) and len(v) > 200)
                }
                if extra:
                    msg += "\n" + json.dumps(extra, ensure_ascii=False)[:500]
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"Change3D call failed: {e}"

    def _call_sm3det(self, tool_name: str, args: Dict[str, Any], api_url: str) -> str:
        task = SM3DET_TOOL_TO_TASK.get(tool_name, "oriented_detection")
        image_path = args.get("image_path") or ""
        classes = args.get("classes") or []

        if not image_path:
            return "Error: Missing image_path."

        try:
            payload = {
                "task": task,
                "image": encode_image(image_path),
                "text": "",
                "classes": classes,
                "box_threshold": 0.3,
                "modality": args.get("modality", "rgb"),
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_default)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                if "detections" in data:
                    total = sum(
                        len(v) for v in (data["detections"] or {}).values() if isinstance(v, list)
                    )
                    msg += f"\nDetected {total} objects."
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"SM3Det call failed: {e}"

    def _call_skysense_det(self, args: Dict[str, Any], api_url: str) -> str:
        image_path = args.get("image_path") or ""
        if not image_path:
            return "Error: Missing image_path."

        try:
            payload = {
                "task": SKYSENSE_DET_TASK,
                "image": encode_image(image_path),
                "score_thr": 0.3,
                "classes": args.get("classes"),
                "return_image": True,
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_default)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                if "num_detections" in data:
                    msg += f"\nDetected {data['num_detections']} objects."
                if "detections" in data and data["detections"]:
                    msg += "\nClasses: " + ", ".join(data["detections"].keys())
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"SkySense Detection call failed: {e}"

    def _call_crossearth(self, tool_name: str, args: Dict[str, Any], api_url: str) -> str:
        task = CROSSEARTH_TOOL_TO_TASK.get(tool_name) or "semantic_seg"
        image_path = args.get("image_path") or ""

        if not image_path:
            return "Error: Missing image_path."

        try:
            payload = {
                "task": task,
                "image": encode_image(image_path),
                "text": args.get("text", ""),
                "classes": args.get("classes") or [],
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_default)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                if "mask" in data and data.get("mask"):
                    enc = data.get("mask_encoding", "")
                    enc_msg = enc or "png_uint16_class_index"
                    msg += f"\nSemantic segmentation completed, mask encoding: {enc_msg}"
                if "classes" in data and data["classes"]:
                    cl = data["classes"]
                    msg += f"\nClasses: {', '.join(cl[:10])}{'...' if len(cl) > 10 else ''}"
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"CrossEarth call failed: {e}"

    def _call_directsam(self, tool_name: str, args: Dict[str, Any], api_url: str) -> str:
        task = tool_name
        image_path = args.get("image_path") or ""

        if not image_path:
            return "Error: Missing image_path."

        try:
            payload = {
                "task": task,
                "image": encode_image(image_path),
                "box": args.get("box"),
                "threshold": args.get("threshold", 0.25),
            }
            data = http_post_json(api_url, payload, timeout=self._timeout_default)
            if data.get("status") == "success":
                msg = data.get("message", "Success")
                if "num_subobjects" in data:
                    msg += f"\nDirectSAM regions count: {data['num_subobjects']}"
                return msg
            return "API Response: " + data.get("message", str(data))
        except Exception as e:
            return f"DirectSAM call failed: {e}"

    @staticmethod
    def execute_tool(
        tool_name: str,
        args: Dict[str, Any],
        api_urls: Dict[str, Optional[str]],
    ) -> str:
        """Backward-compatible helper for benchmarks and scripts."""
        return ServiceExecutor(api_urls).execute(tool_name, args)


execute_tool = ServiceExecutor.execute_tool
