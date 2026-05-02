"""OpenAI-compatible client for vLLM servers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class VLLMChatClient:
    def __init__(self, base_url: str, api_key: str = "dummy") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=f"{self._base_url}/v1", api_key=self._api_key)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 512,
    ) -> str:
        client = self._get_client()
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    @staticmethod
    def list_first_model_id(vllm_url: str, timeout: float = 10.0) -> Optional[str]:
        import json
        import urllib.request

        try:
            req = urllib.request.Request(f"{vllm_url.rstrip('/')}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            models = data.get("data") or []
            if isinstance(models, list) and models:
                return models[0].get("id")
        except Exception:
            pass
        return None
