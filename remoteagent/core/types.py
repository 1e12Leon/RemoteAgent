from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentResult:
    text: str
    history: List[Dict[str, Any]] = field(default_factory=list)
