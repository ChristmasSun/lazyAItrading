from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, UTC, date, timedelta
from zoneinfo import ZoneInfo
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


def persist_decisions(decisions: List[Dict[str, Any]], prices: Dict[str, float], path: str = DECISIONS_LOG, meta: Dict[str, Any] | None = None) -> None:
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
    rec: Dict[str, Any] = {"ts": now_ts(), "decisions": out}
    if meta:
        rec["meta"] = meta
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# --- Market calendar and cadence helpers (duplicated from runner_daily) ---
def _easter_sunday(y: int) -> date:
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def _nth_weekday_of_month(y: int, month: int, weekday: int, n: int) -> date:
    d = date(y, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday_of_month(y: int, month: int, weekday: int) -> date:
    d = date(y, month + 1, 1) - timedelta(days=1) if month < 12 else date(y, 12, 31)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def is_us_market_holiday(d: date) -> bool:
    y = d.year
    new_year = date(y, 1, 1)
    if new_year.weekday() == 5:
        new_year = date(y, 12, 31)
    elif new_year.weekday() == 6:
        new_year = date(y, 1, 2)
    mlk = _nth_weekday_of_month(y, 1, 0, 3)
    presidents = _nth_weekday_of_month(y, 2, 0, 3)
    good_friday = _easter_sunday(y) - timedelta(days=2)
    memorial = _last_weekday_of_month(y, 5, 0)
    juneteenth = date(y, 6, 19)
    if juneteenth.weekday() == 5:
        juneteenth = date(y, 6, 18)
    elif juneteenth.weekday() == 6:
        juneteenth = date(y, 6, 20)
    independence = date(y, 7, 4)
    if independence.weekday() == 5:
        independence = date(y, 7, 3)
    elif independence.weekday() == 6:
        independence = date(y, 7, 5)
    labor = _nth_weekday_of_month(y, 9, 0, 1)
    thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)
    christmas = date(y, 12, 25)
    if christmas.weekday() == 5:
        christmas = date(y, 12, 24)
    elif christmas.weekday() == 6:
        christmas = date(y, 12, 26)
    holidays = {new_year, mlk, presidents, good_friday, memorial, juneteenth, independence, labor, thanksgiving, christmas}
    return d in holidays


def should_act_now(now_ny: datetime) -> bool:
    return now_ny.minute % 15 == 0


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

    # market-hours guard (NY 09:30-16:00, weekdays, 15m gating, holidays)
    try:
        ny = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny)
        wk = now_ny.weekday()
        h = now_ny.hour
        m = now_ny.minute
        open_ok = (wk < 5) and ((h > 9 or (h == 9 and m >= 30)) and (h < 16 or (h == 16 and m == 0)))
        if not open_ok or is_us_market_holiday(now_ny.date()) or not should_act_now(now_ny):
            print(json.dumps({"skipped": True, "reason": "outside_market_hours", "ts": now_ts()}))
            return
    except Exception:
        pass

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
    err = out.get("error") if isinstance(out, dict) else None

    port = Portfolio(cash=cash, fee_rate=args.fee_rate, fee_fixed=args.fee_fixed, slippage_bps=args.slippage_bps, trade_log_path=TRADE_LOG)
    port.positions = positions
    # if no decisions from LLM, build a simple fallback: top by 5-bar momentum (r5d proxy) equal-weighted
    if not decisions:
        parsed: List[Tuple[str, float]] = []
        for row in table_rows:
            try:
                sym, last_px, r1d, r5d, rsi = row.split(",")
                parsed.append((sym, float(r5d)))
            except Exception:
                continue
        parsed.sort(key=lambda x: x[1], reverse=True)
        picks = [sym for sym, _ in parsed[: max(1, args.max_positions)]]
        wt = 1.0 / len(picks) if picks else 0.0
        decisions = [{"symbol": s, "action": "BUY", "target_weight": wt, "reason": "fallback_momentum"} for s in picks]
        persist_decisions(decisions, prices, meta={"fallback": True, "error": err or "empty_decisions"})
    else:
        # log decisions w/ reasons for inspection
        persist_decisions(decisions, prices)
    apply_decisions(port, prices, decisions, args.max_positions, args.min_trade_cash_pct)
    eq = port.value(prices)
    save_state(port.cash, port.positions)
    persist_equity(eq)
    print(json.dumps({"equity": eq, "decisions": decisions[:3], "ts": now_ts()}))


if __name__ == "__main__":
    main()
