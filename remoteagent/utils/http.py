from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict


class HttpUtils:
    @staticmethod
    def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
        url = url.rstrip("/")
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


http_post_json = HttpUtils.http_post_json
