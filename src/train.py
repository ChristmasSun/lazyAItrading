from __future__ import annotations

import argparse
import json
import os
from typing import List
import importlib

from .data.fetch import fetch_ohlcv
from .data.features import build_features_advanced
from .train_torch import train_torch


def train(symbols: List[str], days: int, out: str, interval: str = "1d", period: str | None = None) -> None:
    # collect data
    all_X = []
    all_y = []
    for sym in symbols:
        series = fetch_ohlcv(sym, interval=interval, period=period)
        feats = build_features_advanced(series, lookback=min(120, len(series)))
        X = feats["X"]
        # simple next-day return sign label (stub)
        y = [0] * len(X)
        all_X.extend(X)
        all_y.extend(y)

    # real torch train with checkpoint-style artifact (JSON + .pt)
    out_no_ext = os.path.splitext(out)[0]
    train_torch(symbols, out_no_ext, max_days=days, epochs=10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["AAPL"]) 
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--out", type=str, default="artifacts/models/tiny.json")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--period", type=str, default=None)
    args = p.parse_args()
    train(args.symbols, args.days, args.out, interval=args.interval, period=args.period)


if __name__ == "__main__":
    main()
