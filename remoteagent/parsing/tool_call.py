import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from remoteagent.config.tools_schema import CANONICAL_TO_PROMPT, MCP_TOOLS, PROMPT_TOOL_ALIAS
from remoteagent.utils.text import strip_leading_think_block


class ToolCallParser:
    @staticmethod
    def _parse_path_and_prompt(args_str: str) -> Optional[Dict[str, Any]]:
        m = re.match(r'["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']', args_str)
        if m:
            return {"image_path": m.group(1).strip(), "prompt": m.group(2).strip()}
        return None

    @staticmethod
    def _parse_path_and_classes(args_str: str) -> Optional[Dict[str, Any]]:
        path_m = re.match(r'["\']([^"\']*)["\']\s*,\s*\[(.*)\]', args_str, re.DOTALL)
        if path_m:
            list_part = path_m.group(2)
            classes = re.findall(r'["\']([^"\']*)["\']', list_part)
            cls = [c.strip() for c in classes if c.strip()]
            return {"image_path": path_m.group(1).strip(), "classes": cls}
        return None

    @staticmethod
    def _parse_keyword_path_and_classes(args_str: str) -> Optional[Dict[str, Any]]:
        path_m = re.search(r'(?:image_path|image)\s*=\s*["\']([^"\']*)["\']', args_str)
        classes_m = re.search(r'classes\s*=\s*\[(.*?)\]', args_str, re.DOTALL)
        if path_m and classes_m:
            classes = re.findall(r'["\']([^"\']*)["\']', classes_m.group(1))
            cls = [c.strip() for c in classes if c.strip()]
            return {"image_path": path_m.group(1).strip(), "classes": cls}
        return None

    @staticmethod
    def _parse_classes_only(args_str: str) -> Optional[Dict[str, Any]]:
        s = args_str.strip()
        if not s.startswith("["):
            return None
        classes = re.findall(r'["\']([^"\']*)["\']', s)
        if classes:
            return {"classes": [c.strip() for c in classes if c.strip()]}
        return None

    @staticmethod
    def _parse_path_only(args_str: str) -> Optional[Dict[str, Any]]:
        m = re.match(r'["\']([^"\']*)["\']', args_str.strip())
        if m:
            return {"image_path": m.group(1).strip()}
        return None

    @staticmethod
    def _parse_two_paths(args_str: str) -> Optional[Dict[str, Any]]:
        m = re.match(r'["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']', args_str)
        if m:
            return {"pre_image_path": m.group(1).strip(), "post_image_path": m.group(2).strip()}
        return None

    @staticmethod
    def _parse_path_classes_modality(args_str: str) -> Optional[Dict[str, Any]]:
        path_m2 = re.match(r'["\']([^"\']*)["\']\s*,\s*\[(.*)\]', args_str, re.DOTALL)
        if path_m2:
            list_part = path_m2.group(2)
            classes = re.findall(r'["\']([^"\']*)["\']', list_part)
            out: Dict[str, Any] = {
                "image_path": path_m2.group(1).strip(),
                "classes": [c.strip() for c in classes if c.strip()],
            }
            mod = re.search(r'modality\s*=\s*["\']?(\w+)["\']?', args_str)
            if mod:
                out["modality"] = mod.group(1).strip()
            return out
        return None

    @staticmethod
    def _parse_path_box_threshold(args_str: str) -> Optional[Dict[str, Any]]:
        path_box = re.match(
            r'["\']([^"\']*)["\']\s*,\s*\[(.*?)\](?:\s*,\s*threshold\s*=\s*([\d.]+))?',
            args_str,
            re.DOTALL,
        )
        if path_box:
            box_str = path_box.group(2)
            nums = re.findall(r'[\d.]+', box_str)
            box = [float(x) for x in nums[:4]] if len(nums) >= 4 else None
            if box is not None:
                out: Dict[str, Any] = {"image_path": path_box.group(1).strip(), "box": box}
                if path_box.group(3):
                    out["threshold"] = float(path_box.group(3))
                return out

        path_m = re.search(r'(?:image_path|image)\s*=\s*["\']([^"\']*)["\']', args_str)
        box_m = re.search(r'(?:region|box)\s*=\s*\[(.*?)\]', args_str, re.DOTALL)
        if path_m and box_m:
            nums = re.findall(r'[\d.]+', box_m.group(1))
            box = [float(x) for x in nums[:4]] if len(nums) >= 4 else None
            if box is not None:
                out = {"image_path": path_m.group(1).strip(), "box": box}
                thr = re.search(r'threshold\s*=\s*([\d.]+)', args_str)
                if thr:
                    out["threshold"] = float(thr.group(1))
                return out
        return None

    @staticmethod
    def _parse_path_threshold(args_str: str) -> Optional[Dict[str, Any]]:
        m = re.match(r'["\']([^"\']*)["\'](?:\s*,\s*threshold\s*=\s*([\d.]+))?', args_str.strip())
        if m:
            out: Dict[str, Any] = {"image_path": m.group(1).strip()}
            if m.group(2):
                out["threshold"] = float(m.group(2))
            return out
        return None

    @staticmethod
    def _parse_keyword_path_and_text(args_str: str) -> Optional[Dict[str, Any]]:
        path_m = re.search(r'(?:image_path|image)\s*=\s*["\']([^"\']*)["\']', args_str)
        text_m = re.search(r'(?:text|prompt)\s*=\s*["\']([^"\']*)["\']', args_str)
        if path_m and text_m:
            return {"image_path": path_m.group(1).strip(), "prompt": text_m.group(1).strip()}
        return None

    def _parse_t_call(self, raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        m = re.match(r'T_call\s*\(\s*([\w_]+)\s*,\s*(.+)\s*\)\s*$', raw.strip(), re.DOTALL)
        if not m:
            return None

        tool_name_raw = m.group(1).strip()
        args_part = m.group(2).strip()
        canonical = PROMPT_TOOL_ALIAS.get(tool_name_raw, tool_name_raw)

        if canonical not in MCP_TOOLS:
            return None

        parsed: Optional[Dict[str, Any]] = None

        if canonical == "referring_expression_segmentation":
            parsed = self._parse_path_and_prompt(args_part) or self._parse_keyword_path_and_text(args_part)
        elif canonical in (
            "skysense_detection",
            "semantic_segmentation",
            "crossearth_semantic_segmentation",
        ):
            parsed = (
                self._parse_path_and_classes(args_part)
                or self._parse_keyword_path_and_classes(args_part)
                or self._parse_classes_only(args_part)
            )
        elif canonical in ("change3d_bcd", "change3d_scd", "change3d_bda"):
            parsed = self._parse_two_paths(args_part)
        elif canonical == "sm3det_oriented_detection":
            parsed = self._parse_path_classes_modality(args_part)
        elif canonical in ("contour_extraction", "subobject_contour_extraction"):
            parsed = self._parse_path_threshold(args_part) or self._parse_path_only(args_part)
        elif canonical in ("region_contour_extraction", "region_subobject_contour_extraction"):
            parsed = self._parse_path_box_threshold(args_part)

        if parsed:
            return (canonical, parsed)
        return None

    def parse_tool_call(self, raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not raw or not isinstance(raw, str):
            return None

        body = strip_leading_think_block(raw).strip()
        if not body:
            return None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("T_call"):
                continue
            result = self._parse_t_call(line)
            if result:
                return result
        return None

    def parse_tool_call_with_image_fallback(
        self, raw: str, image_path: Optional[Path] = None
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        result = self.parse_tool_call(raw)
        if not result:
            return None

        tool_name, args = result
        path_str = str(image_path) if image_path else None

        if path_str and "image_path" not in args:
            required = MCP_TOOLS.get(tool_name, [])
            if "image_path" in required:
                args = dict(args)
                args["image_path"] = path_str
                return (tool_name, args)

        return result

    def parse_tool_call_with_bcd_fallback(
        self,
        raw: str,
        pre_image_path: Optional[Path] = None,
        post_image_path: Optional[Path] = None,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        result = self.parse_tool_call(raw)
        if not result:
            return None

        tool_name, args = result
        if tool_name not in ("change3d_bcd", "change3d_scd", "change3d_bda"):
            return result

        args = dict(args)
        if pre_image_path and "pre_image_path" not in args:
            args["pre_image_path"] = str(pre_image_path)
        if post_image_path and "post_image_path" not in args:
            args["post_image_path"] = str(post_image_path)

        return (tool_name, args)

    def format_tool_call_for_display(self, tool_name: str, args: Dict[str, Any]) -> str:
        display_name = CANONICAL_TO_PROMPT.get(tool_name, tool_name)

        if tool_name in ("skysense_detection", "semantic_segmentation", "crossearth_semantic_segmentation"):
            path = args.get("image_path", "")
            classes = args.get("classes", [])
            return f'T_call({display_name}, "{path}", {classes})'

        if tool_name == "referring_expression_segmentation":
            path = args.get("image_path", "")
            prompt = args.get("prompt", "")
            return f'T_call({display_name}, "{path}", "{prompt}")'

        return f"T_call({display_name}, {args})"


DEFAULT_TOOL_CALL_PARSER = ToolCallParser()

parse_tool_call = DEFAULT_TOOL_CALL_PARSER.parse_tool_call
parse_tool_call_with_image_fallback = DEFAULT_TOOL_CALL_PARSER.parse_tool_call_with_image_fallback
parse_tool_call_with_bcd_fallback = DEFAULT_TOOL_CALL_PARSER.parse_tool_call_with_bcd_fallback
format_tool_call_for_display = DEFAULT_TOOL_CALL_PARSER.format_tool_call_for_display
