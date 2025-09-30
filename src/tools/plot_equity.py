from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple


def read_equity(path: str) -> List[Tuple[datetime, float]]:
    pts: List[Tuple[datetime, float]] = []
    if not os.path.exists(path):
        return pts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ts = datetime.fromisoformat(rec.get("ts"))
                eq = float(rec.get("equity", 0.0))
                pts.append((ts, eq))
            except Exception:
                continue
    pts.sort(key=lambda x: x[0])
    return pts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--equity", type=str, default="artifacts/equity.jsonl")
    p.add_argument("--out", type=str, default="artifacts/equity.png")
    args = p.parse_args()

    pts = read_equity(args.equity)
    if not pts:
        print("no equity points to plot")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="equity")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
