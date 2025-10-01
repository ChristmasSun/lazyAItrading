from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple


def read_equity(path: str) -> List[Tuple[datetime, float]]:
    pts: List[Tuple[datetime, float]] = []
    if not os.path.exists(path):
        return pts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                tstr = rec.get("ts")
                if not tstr:
                    continue
                # handle 'Z' UTC suffix from shell bootstrap: convert to +00:00 for fromisoformat
                if isinstance(tstr, str) and tstr.endswith("Z"):
                    tstr = tstr.replace("Z", "+00:00")
                ts = datetime.fromisoformat(tstr)
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
        from matplotlib.ticker import FuncFormatter
    except Exception:
        print("matplotlib not available")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # if only one point, add a small +1 minute offset so a flat line segment is visible
    if len(xs) == 1:
        xs = [xs[0], xs[0] + timedelta(minutes=1)]
        ys = [ys[0], ys[0]]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="equity", marker="o", markersize=3, linewidth=1.6)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    # hide scientific notation offset just in case
    try:
        ax.get_yaxis().get_offset_text().set_visible(False)
    except Exception:
        pass
    # set tight y-limits to avoid confusing small tick labels like -2, 0, 2 with a large offset
    if ys:
        y_min = min(ys)
        y_max = max(ys)
        if y_min == y_max:
            pad = max(1.0, abs(y_min) * 0.01)
        else:
            pad = max(1.0, (y_max - y_min) * 0.05)
        ax.set_ylim(y_min - pad, y_max + pad)
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
