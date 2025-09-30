from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, UTC
from typing import Any, Dict, List, Tuple

from .data.fetch import fetch_ohlcv
from .backtest.portfolio import Portfolio, Position
from .tools.universe import load_universe
from .llm.gemini_client import call_gemini_json, build_prompt


STATE_PATH = "artifacts_gemini/state/portfolio.json"
EQUITY_LOG = "artifacts_gemini/equity.jsonl"
TRADE_LOG = "artifacts_gemini/trades.jsonl"
DECISIONS_LOG = "artifacts_gemini/decisions.jsonl"


def now_ts() -> str:
    return datetime.now(UTC).isoformat()


def load_state(path: str = STATE_PATH) -> Tuple[float, Dict[str, Position]]:
    if not os.path.exists(path):
        return 10_000.0, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cash = float(data.get("cash", 10_000.0))
        positions: Dict[str, Position] = {}
        for sym, pd in data.get("positions", {}).items():
            positions[sym] = Position(qty=float(pd.get("qty", 0.0)), avg_price=float(pd.get("avg_price", 0.0)))
        return cash, positions
    except Exception:
        return 10_000.0, {}


def save_state(cash: float, positions: Dict[str, Position], path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "cash": cash,
        "positions": {sym: {"qty": pos.qty, "avg_price": pos.avg_price} for sym, pos in positions.items()},
        "ts": now_ts(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def persist_equity(value: float, path: str = EQUITY_LOG) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_ts(), "equity": value}) + "\n")


def persist_decisions(decisions: List[Dict[str, Any]], prices: Dict[str, float], path: str = DECISIONS_LOG) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # enrich with last_price and keep only relevant fields
    out: List[Dict[str, Any]] = []
    for d in decisions:
        sym = str(d.get("symbol", "")).upper()
        out.append({
            "symbol": sym,
            "action": (d.get("action") or "").upper(),
            "target_weight": float(d.get("target_weight", 0.0)),
            "reason": d.get("reason", ""),
            "last_price": float(prices.get(sym, 0.0)),
        })
    rec = {"ts": now_ts(), "decisions": out}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def summarize(symbols: List[str], interval: str, period: str) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    ohlcv_map: Dict[str, List[Dict[str, Any]]] = {}
    rows: List[str] = []
    for s in symbols:
        series = fetch_ohlcv(s, interval=interval, period=period)
        ohlcv_map[s] = series
        if len(series) >= 6:
            px = series[-1]["close"]
            r1 = (series[-1]["close"] - series[-2]["close"]) / (series[-2]["close"] or 1)
            r5 = (series[-1]["close"] - series[-6]["close"]) / (series[-6]["close"] or 1)
            # naive RSI approx using last 14 changes
            diffs = [series[i]["close"] - series[i - 1]["close"] for i in range(-13, 0)] if len(series) >= 15 else [0.0]
            gains = sum(d for d in diffs if d > 0) / max(1, len(diffs))
            losses = sum(-d for d in diffs if d < 0) / max(1, len(diffs))
            rs = gains / (losses or 1)
            rsi = 100 - (100 / (1 + rs))
            rows.append(f"{s},{px:.4f},{r1:.5f},{r5:.5f},{rsi:.2f}")
    return ohlcv_map, rows


def apply_decisions(port: Portfolio, prices: Dict[str, float], decisions: List[Dict[str, Any]], max_positions: int, min_trade_cash_pct: float) -> None:
    # convert decisions to target weights
    # keep only up to max_positions and normalize weights
    picks: List[Tuple[str, float]] = []
    for d in decisions:
        sym = str(d.get("symbol", "")).upper()
        act = (d.get("action", "HOLD") or "").upper()
        tw = float(d.get("target_weight", 0.0))
        if sym and act in ("BUY", "HOLD") and tw > 0:
            picks.append((sym, tw))
    picks = picks[:max_positions]
    total = sum(w for _, w in picks) or 1.0
    targets = {sym: w / total for sym, w in picks}

    eq = port.value(prices)
    thresh = max(min_trade_cash_pct * eq, 1.0)

    # sell anything not targeted
    for sym, pos in list(port.positions.items()):
        if pos.qty <= 0:
            continue
        if sym not in targets:
            px = prices.get(sym, 0.0)
            if px > 0:
                port.sell(sym, px, pos.qty)

    # adjust to targets
    eq = port.value(prices)
    for sym, tw in targets.items():
        px = prices.get(sym, 0.0)
        if px <= 0:
            continue
        pos = port.positions.get(sym)
        cur_val = (pos.qty * px) if pos else 0.0
        tgt_val = tw * eq
        delta = tgt_val - cur_val
        if abs(delta) < thresh:
            continue
        if delta > 0:
            port.buy(sym, px, delta)
        else:
            sell_qty = min(pos.qty if pos else 0.0, abs(delta) / px)
            if sell_qty > 0:
                port.sell(sym, px, sell_qty)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=str, default="sp500")
    ap.add_argument("--max-symbols", type=int, default=500)
    ap.add_argument("--interval", type=str, default="5m")
    ap.add_argument("--period", type=str, default="5d")
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--fee-fixed", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--min-trade-cash-pct", type=float, default=0.002)
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--model", type=str, default="gemini-1.5-pro-latest")
    ap.add_argument("--temperature", type=float, default=0.4)
    args = ap.parse_args()

    syms = load_universe(args.universe)[: args.max_symbols]
    ohlcv_map, table_rows = summarize(syms, args.interval, args.period)

    # build portfolio snapshot for prompt
    cash, positions = load_state()
    prices = {s: (ohlcv_map[s][-1]["close"] if ohlcv_map.get(s) else 0.0) for s in syms}
    pf = {
        "cash": cash,
        "positions": {sym: {"qty": pos.qty, "avg_price": pos.avg_price} for sym, pos in positions.items()},
    }
    rules = {
        "objective": "maximize equity",
        "max_positions": args.max_positions,
        "notes": "focus on liquid S&P names, avoid overtrading",
    }

    prompt = build_prompt(pf, table_rows, rules, allowed_symbols=syms)
    out = call_gemini_json(prompt, model_name=args.model, temperature=args.temperature)
    decisions = out.get("decisions", []) if isinstance(out, dict) else []

    port = Portfolio(cash=cash, fee_rate=args.fee_rate, fee_fixed=args.fee_fixed, slippage_bps=args.slippage_bps, trade_log_path=TRADE_LOG)
    port.positions = positions
    # log decisions w/ reasons for inspection
    persist_decisions(decisions, prices)
    apply_decisions(port, prices, decisions, args.max_positions, args.min_trade_cash_pct)
    eq = port.value(prices)
    save_state(port.cash, port.positions)
    persist_equity(eq)
    print(json.dumps({"equity": eq, "decisions": decisions[:3], "ts": now_ts()}))


if __name__ == "__main__":
    main()
