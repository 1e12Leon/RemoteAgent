from __future__ import annotations

import base64
import os


class ImageUtils:
    @staticmethod
    def encode_image(path: str) -> str:
        path = path.strip().strip("'").strip('"')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


encode_image = ImageUtils.encode_image
