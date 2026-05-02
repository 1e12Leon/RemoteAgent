"""Default service ports and environment variable names."""

from __future__ import annotations

from typing import Dict

SERVICE_PORTS: Dict[str, int] = {
    "remotesam": 6657,
    "change3d": 6658,
    "sm3det": 6655,
    "crossearth": 6656,
    "skysense_det": 6654,
    "directsam": 6659,
}

ENV_URL_KEYS: Dict[str, str] = {
    "remotesam": "REMOTE_API_URL",
    "change3d": "CHANGE3D_API_URL",
    "sm3det": "SM3DET_API_URL",
    "crossearth": "CROSSEARTH_API_URL",
    "skysense_det": "SKYSENSE_DET_API_URL",
    "directsam": "DIRECTSAM_API_URL",
}

DEFAULT_VLLM_URL = "http://localhost:8000"
