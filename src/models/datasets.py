from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class OHLCVBar:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


def make_dummy_series(n: int = 60) -> List[Dict[str, Any]]:
    """tiny synthetic series for quick tests"""
    base = 100.0
    out: List[Dict[str, Any]] = []
    for i in range(n):
        # toy trend
        price = base + i * 0.1
        out.append({"ts": i, "open": price, "high": price + 0.2, "low": price - 0.2, "close": price, "volume": 1000 + i})
    return out
