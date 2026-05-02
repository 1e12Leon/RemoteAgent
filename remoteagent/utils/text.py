from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_THINK_LEADING_RE = re.compile(
    r"^\s*<think>\s*[\s\S]*?\s*</think>\s*",
    re.IGNORECASE,
)


class TextUtils:
    @staticmethod
    def strip_leading_think_block(response: str) -> str:
        """Strip one or more leading <think>...</think> blocks."""
        if not response or not isinstance(response, str):
            return response
        out = response
        while True:
            nxt = _THINK_LEADING_RE.sub("", out, count=1)
            if nxt == out:
                return out
            out = nxt

    @staticmethod
    def extract_answer_tag(response: str) -> Optional[str]:
        if not response or not isinstance(response, str):
            return None
        body = TextUtils.strip_leading_think_block(response).strip()
        m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", body, re.IGNORECASE)
        if not m:
            return None
        return m.group(1).strip()

    @staticmethod
    def load_system_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()


strip_leading_think_block = TextUtils.strip_leading_think_block
extract_answer_tag = TextUtils.extract_answer_tag
load_system_prompt = TextUtils.load_system_prompt
