from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, UTC, date, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Tuple

from .tools.universe import load_universe
from .data.fetch import fetch_ohlcv
from .backtest.portfolio import Portfolio, Position
from .agents.risk import RiskAgent


STATE_PATH = "artifacts/state/portfolio.json"
EQUITY_LOG = "artifacts/equity.jsonl"
PICKS_DIR = "artifacts/picks"
TRADE_LOG = "artifacts/trades/trades.jsonl"


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


def persist_equity_point(value: float, path: str = EQUITY_LOG) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_ts(), "equity": value}) + "\n")


def latest_picks_file(directory: str = PICKS_DIR) -> str | None:
    if not os.path.isdir(directory):
        return None
    files = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(".csv")]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def read_picks(path: str) -> List[Tuple[str, float, float]]:
    """Return list of (symbol, score, weight). alloc is optional and ignored here for targets."""
    out: List[Tuple[str, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        cr = csv.DictReader(f)
        for row in cr:
            try:
                sym = row.get("symbol", "").strip()
                sc = float(row.get("score", 0.0))
                w = float(row.get("weight", 0.0))
                if sym:
                    out.append((sym, sc, w))
            except Exception:
                continue
    return out


def fetch_prices(symbols: List[str], interval: str = "1d", period: str = "1mo") -> Dict[str, float]:
    prices: Dict[str, float] = {}
    # basic throttle to reduce rate pressure
    import os, time
    sleep_s = float(os.environ.get("FETCH_SLEEP_S", "0.05"))
    for s in symbols:
        series = fetch_ohlcv(s, interval=interval, period=period)
        px = float(series[-1]["close"]) if series else 0.0
        prices[s] = px
        if sleep_s > 0:
            time.sleep(sleep_s)
    return prices


# --- Market calendar and cadence helpers ---
def _easter_sunday(y: int) -> date:
    # Anonymous Gregorian algorithm
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
    # weekday: Mon=0 .. Sun=6
    d = date(y, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday_of_month(y: int, month: int, weekday: int) -> date:
    d = date(y, month + 1, 1) - timedelta(days=1) if month < 12 else date(y, 12, 31)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def is_us_market_holiday(d: date) -> bool:
    y = d.year
    # New Year's Day (observed)
    new_year = date(y, 1, 1)
    if new_year.weekday() == 5:  # Sat
        new_year = date(y, 12, 31) if y > 1900 else new_year
    elif new_year.weekday() == 6:  # Sun
        new_year = date(y, 1, 2)

    mlk = _nth_weekday_of_month(y, 1, 0, 3)  # 3rd Mon Jan
    presidents = _nth_weekday_of_month(y, 2, 0, 3)  # 3rd Mon Feb
    good_friday = _easter_sunday(y) - timedelta(days=2)
    memorial = _last_weekday_of_month(y, 5, 0)  # last Mon May
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
    labor = _nth_weekday_of_month(y, 9, 0, 1)  # 1st Mon Sep
    thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)  # 4th Thu Nov
    christmas = date(y, 12, 25)
    if christmas.weekday() == 5:
        christmas = date(y, 12, 24)
    elif christmas.weekday() == 6:
        christmas = date(y, 12, 26)

    holidays = {new_year, mlk, presidents, good_friday, memorial, juneteenth, independence, labor, thanksgiving, christmas}
    return d in holidays


def should_act_now(now_ny: datetime) -> bool:
    # only on :00 :15 :30 :45
    return now_ny.minute % 15 == 0


def should_refresh_picks(now_ny: datetime, latest_picks_path: str | None, staleness_minutes: int = 60) -> bool:
    # refresh at top of hour or if no picks file or stale
    if latest_picks_path is None or not os.path.exists(latest_picks_path):
        return True
    if now_ny.minute == 0:
        return True
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(latest_picks_path), tz=now_ny.tzinfo)
        age = now_ny - mtime
        return age.total_seconds() >= staleness_minutes * 60
    except Exception:
        return True


def rebalance_to_picks(
    picks: List[Tuple[str, float, float]],
    fee_rate: float,
    fee_fixed: float,
    slippage_bps: float,
    min_trade_cash_pct: float,
    profile: str,
    interval: str,
    period: str,
) -> float:
    # load state -> build portfolio -> compute targets
    cash, positions = load_state()
    port = Portfolio(cash=cash, fee_rate=fee_rate, fee_fixed=fee_fixed, slippage_bps=slippage_bps, trade_log_path=TRADE_LOG)
    port.positions = positions

    # build symbol set and prices
    target_syms = [sym for sym, _, _ in picks]
    held_syms = list(positions.keys())
    all_syms = list({*target_syms, *held_syms})
    prices = fetch_prices(all_syms, interval=interval, period=period)

    # risk cap and stop-loss
    cap = RiskAgent.PROFILES.get(profile, RiskAgent.PROFILES["balanced"]).max_position_pct
    stop_loss_pct = RiskAgent.PROFILES.get(profile, RiskAgent.PROFILES["balanced"]).stop_loss_pct
    # normalize weights and cap
    total_w = sum(max(0.0, w) for _, _, w in picks)
    norm = 1.0 / total_w if total_w > 0 else 0.0
    targets: Dict[str, float] = {}
    for sym, _, w in picks:
        ww = max(0.0, w) * norm if norm > 0 else 0.0
        targets[sym] = min(ww, cap)
    # scale down if sum > 1
    sum_w = sum(targets.values())
    if sum_w > 1.0:
        scale = 1.0 / sum_w
        for k in list(targets.keys()):
            targets[k] *= scale

    eq = port.value(prices)
    thresh = max(min_trade_cash_pct * eq, 1.0)

    # enforce stop-loss before target rebalance
    for sym in held_syms:
        pos = port.positions.get(sym)
        px = prices.get(sym, 0.0)
        if pos and pos.qty > 0 and px > 0 and pos.avg_price > 0:
            if px <= pos.avg_price * (1.0 - stop_loss_pct):
                port.sell(sym, px, pos.qty)

    # sell names not in picks to target 0
    for sym in held_syms:
        if sym not in targets:
            px = prices.get(sym, 0.0)
            pos = port.positions.get(sym)
            if pos and pos.qty > 0 and px > 0:
                port.sell(sym, px, pos.qty)

    # adjust picks to target values
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

    # persist state and return equity
    final_eq = port.value(prices)
    save_state(port.cash, port.positions)
    return final_eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-picks", action="store_true", help="if set, read latest picks CSV and rebalance to it; else fallback")
    ap.add_argument("--autopilot", action="store_true", help="run predict -> write picks -> rebalance in one go")
    ap.add_argument("--picks-file", type=str, default="", help="explicit path to picks CSV; overrides latest selection")
    ap.add_argument("--universe", type=str, default="sp500")
    ap.add_argument("--max-symbols", type=int, default=500)
    ap.add_argument("--interval", type=str, default="5m")
    ap.add_argument("--period", type=str, default="5d")
    ap.add_argument("--cash", type=float, default=10_000.0, help="initial cash if no state exists")
    ap.add_argument("--profile", type=str, default="balanced")
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--fee-fixed", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--min-trade-cash-pct", type=float, default=0.002)
    ap.add_argument("--market-hours-only", action="store_true", help="if set, do nothing outside US market hours (Mon-Fri 09:30-16:00 ET)")
    args = ap.parse_args()

    # bootstrap state if missing
    if not os.path.exists(STATE_PATH):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        save_state(args.cash, {})

    # market hours guard
    if args.market_hours_only:
        try:
            ny = ZoneInfo("America/New_York")
            now_ny = datetime.now(ny)
            # weekday() 0=Mon..6=Sun
            wk = now_ny.weekday()
            h = now_ny.hour
            m = now_ny.minute
            open_ok = (wk < 5) and ((h > 9 or (h == 9 and m >= 30)) and (h < 16 or (h == 16 and m == 0)))
            if not open_ok or is_us_market_holiday(now_ny.date()) or not should_act_now(now_ny):
                print(json.dumps({"skipped": True, "reason": "outside_market_hours", "ts": now_ts()}))
                return
        except Exception:
            # if timezone fails just proceed
            pass

    # optional autopilot: generate picks first
    if args.autopilot:
        from .predict_cli import main as predict_main  # local import to avoid overhead when not used
        latest_path = latest_picks_file()
        ny = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny)
        if should_refresh_picks(now_ny, latest_path):
            # create a fresh picks CSV using universe
            predict_args = [
                "--universe", args.universe,
                "--max-symbols", str(args.max_symbols),
                "--top-n", str(min(args.max_symbols, 20)),
                "--weights", "score",
                "--profile", args.profile,
                "--cash", "0",
                "--interval", args.interval,
                "--period", args.period,
            ]
            import sys
            sys.argv = ["predict_cli"] + predict_args
            try:
                predict_main()
            except SystemExit:
                pass

    if args.use_picks or args.picks_file:
        pfile = args.picks_file or latest_picks_file()
        if pfile and os.path.exists(pfile):
            picks = read_picks(pfile)
            final_eq = rebalance_to_picks(
                picks,
                fee_rate=args.fee_rate,
                fee_fixed=args.fee_fixed,
                slippage_bps=args.slippage_bps,
                min_trade_cash_pct=args.min_trade_cash_pct,
                profile=args.profile,
                interval=args.interval,
                period=args.period,
            )
            persist_equity_point(final_eq)
            print(json.dumps({"mode": "picks", "equity": final_eq, "picks_file": pfile}))
            return

    # fallback: no picks -> just score a slice of the universe and simulate a single rebalance via backtest-like step
    symbols = load_universe(args.universe)[: args.max_symbols]
    # build a fake equal-weight picks list
    eq_w = 1.0 / max(1, len(symbols))
    picks = [(s, 0.0, eq_w) for s in symbols]
    final_eq = rebalance_to_picks(
        picks,
        fee_rate=args.fee_rate,
        fee_fixed=args.fee_fixed,
        slippage_bps=args.slippage_bps,
        min_trade_cash_pct=args.min_trade_cash_pct,
        profile=args.profile,
        interval=args.interval,
        period=args.period,
    )
    persist_equity_point(final_eq)
    print(json.dumps({"mode": "fallback", "equity": final_eq}))


if __name__ == "__main__":
    main()
