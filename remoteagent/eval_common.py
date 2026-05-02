"""Shared helpers for benchmark scripts under ``tasks/eval/``."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from remoteagent.utils.image import encode_image

UserContent = Union[str, List[Dict[str, Any]]]


class EvalCommon:
    @staticmethod
    def http_post_predict(
        api_url: str, payload: Dict[str, Any], timeout: float = 120.0
    ) -> Dict[str, Any]:
        """POST JSON to a tool server ``/predict`` endpoint."""
        url = api_url.rstrip("/")
        if not url.endswith("/predict"):
            url = f"{url}/predict"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def get_vllm_model_id(api_url: str, timeout: float = 10.0) -> Optional[str]:
        """Return first model id from vLLM ``/v1/models``."""
        try:
            req = urllib.request.Request(f"{api_url.rstrip('/')}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                models = json.loads(r.read().decode()).get("data")
            if isinstance(models, list) and models and isinstance(models[0], dict):
                return models[0].get("id")
        except Exception:
            pass
        return None

    @staticmethod
    def load_eval_system_prompt(eval_dir: Path, path: Optional[Path] = None) -> str:
        """Load ``prompt.txt`` from *eval_dir* unless *path* is provided and exists."""
        p = path if path and path.exists() else eval_dir / "prompt.txt"
        if not p.exists():
            raise FileNotFoundError(f"System prompt not found: {p}")
        return p.read_text(encoding="utf-8").strip()

    @staticmethod
    def run_vllm_chat(
        system_prompt: str,
        user_prompt: str,
        api_url: str,
        model_name: str,
        *,
        image_path: Optional[Path] = None,
        image_paths: Optional[List[Path]] = None,
        max_tokens: int = 512,
    ) -> str:
        """OpenAI-compatible chat with optional one or two images."""
        try:
            from openai import OpenAI

            client = OpenAI(base_url=f"{api_url.rstrip('/')}/v1", api_key="dummy")

            if image_paths and len(image_paths) >= 2:
                parts: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
                for p in image_paths[:2]:
                    if p.exists():
                        img_b64 = encode_image(str(p))
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                            }
                        )
                user_content: UserContent = parts
            elif image_path and image_path.exists():
                img_b64 = encode_image(str(image_path))
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]
            else:
                user_content = user_prompt

            r = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"[ERROR:{e}]"


http_post_predict = EvalCommon.http_post_predict
get_vllm_model_id = EvalCommon.get_vllm_model_id
load_eval_system_prompt = EvalCommon.load_eval_system_prompt
run_vllm_chat = EvalCommon.run_vllm_chat
