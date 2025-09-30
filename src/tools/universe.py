from __future__ import annotations

import csv
import os
from typing import List


UNIVERSE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "universes")
os.makedirs(UNIVERSE_DIR, exist_ok=True)


def save_universe(name: str, symbols: List[str]) -> str:
    path = os.path.abspath(os.path.join(UNIVERSE_DIR, f"{name}.csv"))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol"])  # header
        for s in symbols:
            w.writerow([s])
    return path


def load_universe(name: str) -> List[str]:
    path = os.path.abspath(os.path.join(UNIVERSE_DIR, f"{name}.csv"))
    if not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("symbol") or "").strip()
            if s:
                out.append(s)
    return out
