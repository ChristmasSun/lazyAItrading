from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List


def _cache_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "data", "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _key(symbol: str, start: str | None, end: str | None, interval: str) -> str:
    raw = f"{symbol}|{start}|{end}|{interval}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"ohlcv_{h}.json"


def cache_path(symbol: str, start: str | None, end: str | None, interval: str) -> str:
    return os.path.join(_cache_dir(), _key(symbol, start, end, interval))


def is_valid(path: str, ttl_seconds: int) -> bool:
    if not os.path.exists(path):
        return False
    if ttl_seconds <= 0:
        return True
    age = time.time() - os.path.getmtime(path)
    return age <= ttl_seconds


def save(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
