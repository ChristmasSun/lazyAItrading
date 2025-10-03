from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from ..data.fetch import fetch_ohlcv


def read_portfolio(path: str) -> Tuple[float, Dict[str, Dict[str, float]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cash = float(data.get("cash", 0.0))
    positions = {
        str(sym): {
            "qty": float(p.get("qty", 0.0)),
            "avg_price": float(p.get("avg_price", 0.0)),
        }
        for sym, p in (data.get("positions", {}) or {}).items()
    }
    return cash, positions


def latest_prices(symbols: List[str], interval: str, period: str, sleep_s: float = 0.05) -> Tuple[Dict[str, float], Dict[str, str]]:
    prices: Dict[str, float] = {}
    src: Dict[str, str] = {}
    import time
    for s in symbols:
        px = 0.0
        source = "intraday"
        try:
            rows_i = fetch_ohlcv(s, interval=interval, period=period)
        except Exception:
            rows_i = []
        ts_i = rows_i[-1]["ts"] if rows_i else 0
        try:
            rows_d = fetch_ohlcv(s, interval="1d", period="1mo")
        except Exception:
            rows_d = []
        ts_d = rows_d[-1]["ts"] if rows_d else 0
        # choose the freshest bar between intraday and daily
        if ts_d > ts_i and rows_d:
            px = float(rows_d[-1]["close"])
            source = "daily"
        elif rows_i:
            px = float(rows_i[-1]["close"])
            source = "intraday"
        prices[s] = px
        src[s] = source
        if sleep_s > 0:
            time.sleep(sleep_s)
    return prices, src


def compute_value(cash: float, positions: Dict[str, Dict[str, float]], prices: Dict[str, float]) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    pos_val = 0.0
    breakdown: Dict[str, Dict[str, float]] = {}
    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0))
        px = float(prices.get(sym, 0.0))
        if px <= 0.0:
            px = float(p.get("avg_price", 0.0))  # fallback
        val = qty * px
        pos_val += val
        breakdown[sym] = {"qty": qty, "price": px, "value": val}
    equity = cash + pos_val
    return pos_val, equity, breakdown


def read_last_equity(log_path: str) -> float | None:
    if not os.path.exists(log_path):
        return None
    last = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    last = float(rec.get("equity", 0.0))
                except Exception:
                    continue
    except Exception:
        return None
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=str, default="artifacts/state/portfolio.json")
    ap.add_argument("--interval", type=str, default="5m")
    ap.add_argument("--period", type=str, default="5d")
    ap.add_argument("--equity-log", type=str, default="artifacts/equity.jsonl")
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    cash, positions = read_portfolio(args.portfolio)
    syms = sorted(list(positions.keys()))
    prices, src = latest_prices(syms, interval=args.interval, period=args.period, sleep_s=args.sleep)
    pos_val, equity, breakdown = compute_value(cash, positions, prices)

    last_eq = read_last_equity(args.equity_log)
    out: Dict[str, Any] = {
        "cash": cash,
        "positions_count": len(positions),
        "positions_value": pos_val,
        "equity_now": equity,
        "interval": args.interval,
        "period": args.period,
    "breakdown": breakdown,
    "price_source": src,
    }
    if last_eq is not None:
        out["equity_last_log"] = last_eq
        out["delta_vs_last"] = equity - last_eq
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
