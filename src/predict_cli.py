from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from .data.fetch import fetch_ohlcv
from .data.history import save_history_csv, history_path
from .tools.universe import load_universe
from .selection.selector import score_universe
from .agents.risk import RiskAgent
import os
import time


def ensure_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)


def compute_weights(ranked: List[Tuple[str, float]], top_n: int, mode: str = "equal") -> List[Tuple[str, float, float]]:
    picks = ranked[: max(0, top_n)]
    if not picks:
        return []
    if mode == "score":
        scores = [max(0.0, s) for _, s in picks]
        ssum = sum(scores)
        if ssum <= 0:
            w = 1.0 / len(picks)
            return [(sym, sc, w) for sym, sc in picks]
        return [(sym, sc, sc / ssum) for sym, sc in picks]
    # equal
    w = 1.0 / len(picks)
    return [(sym, sc, w) for sym, sc in picks]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"]) 
    p.add_argument("--universe", type=str, default=None)
    p.add_argument("--max-symbols", type=int, default=200)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--weights", type=str, choices=["equal", "score"], default="equal")
    p.add_argument("--interval", type=str, default="5m")
    p.add_argument("--period", type=str, default="5d")
    p.add_argument("--profile", type=str, default="balanced")
    p.add_argument("--cash", type=float, default=0.0, help="if >0, compute dollar allocations respecting max position cap")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    symbols = args.symbols
    if args.universe:
        uni_syms = load_universe(args.universe)
        if uni_syms:
            symbols = uni_syms[: args.max_symbols]

    # throttle settings
    sleep_s = float(os.environ.get("FETCH_SLEEP_S", "0.1"))
    max_fetch = int(os.environ.get("FETCH_MAX", "0"))  # 0 = no cap

    # fetch data and save local CSV history
    ohlcv_map: Dict[str, List[Dict[str, Any]]] = {}
    for idx, s in enumerate(symbols):
        if max_fetch and idx >= max_fetch:
            break
        rows = fetch_ohlcv(s, interval=args.interval, period=args.period)
        ohlcv_map[s] = rows
        save_history_csv(history_path(s, args.interval, args.period), rows)
        if sleep_s > 0:
            time.sleep(sleep_s)

    ranked = score_universe(symbols, ohlcv_map)
    weighted = compute_weights(ranked, args.top_n, mode=args.weights)

    # apply risk-based cap to weights if cash provided
    cap = RiskAgent.PROFILES.get(args.profile, RiskAgent.PROFILES["balanced"]).max_position_pct
    weights = []  # (sym, score, weight, alloc)
    sum_w = 0.0
    for sym, sc, w in weighted:
        w_cap = min(w, cap)
        weights.append((sym, sc, w_cap, 0.0))
        sum_w += w_cap
    # don't scale up if sum < 1 (leave residual cash); if >1, scale down
    scale = 1.0
    if sum_w > 1.0:
        scale = 1.0 / sum_w

    if args.cash > 0:
        weights = [(sym, sc, w, w * scale * args.cash) for sym, sc, w, _ in weights]
    else:
        weights = [(sym, sc, w * scale, 0.0) for sym, sc, w, _ in weights]

    # output
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or os.path.join("artifacts", "picks", f"picks-{ts}.csv")
    ensure_dir(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["symbol", "score", "weight", "alloc_dollars", "last_price", "timestamp"])
        for sym, sc, w, alloc in weights:
            series = ohlcv_map.get(sym, [])
            last_px = float(series[-1]["close"]) if series else 0.0
            wcsv.writerow([sym, f"{sc:.6f}", f"{w:.6f}", f"{alloc:.2f}", f"{last_px:.4f}", ts])

    summary = {
        "universe": args.universe or "custom",
        "symbols_scored": len(symbols),
        "top_n": len(weights),
        "profile": args.profile,
        "cash": args.cash,
        "output": out_path,
        "picks": [
            {"symbol": sym, "score": sc, "weight": w, "alloc_dollars": alloc}
            for sym, sc, w, alloc in weights
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
