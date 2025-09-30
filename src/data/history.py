from __future__ import annotations

import csv
import os
from typing import Any, Dict, List


def history_dir() -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "data", "history")
    os.makedirs(path, exist_ok=True)
    return path


def history_path(symbol: str, interval: str, period: str | None) -> str:
    per = period or "custom"
    fname = f"{symbol}_{interval}_{per}.csv".replace("/", "-")
    return os.path.join(history_dir(), fname)


def save_history_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = ["ts", "open", "high", "low", "close", "volume"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
