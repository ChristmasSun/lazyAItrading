from __future__ import annotations

from typing import Any, Dict


class Trainer:
    """minimal trainer stub to be replaced later with torch impl"""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def fit(self, data: Any) -> None:
        # no-op for now
        return None

    def save(self, path: str) -> None:
        # no-op
        with open(path, "w", encoding="utf-8") as f:
            f.write("stub")
