from __future__ import annotations

import argparse
import json
from typing import Dict, Any, List

from .data.fetch import fetch_ohlcv
from .data.history import save_history_csv, history_path
from .tools.universe import load_universe
from .backtest.engine import backtest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"]) 
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--profile", type=str, default="balanced")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--period", type=str, default="5y")
    p.add_argument("--universe", type=str, default=None)
    p.add_argument("--max-symbols", type=int, default=50)
    p.add_argument("--max-holdings", type=int, default=10)
    p.add_argument("--rebalance-every", type=int, default=0, help="rebalance every N bars; 0=auto from profile")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="proportional fee on notional (e.g., 0.0005 = 5 bps)")
    p.add_argument("--fee-fixed", type=float, default=0.0, help="fixed fee per order")
    p.add_argument("--slippage-bps", type=float, default=2.0, help="slippage in basis points applied to execution prices")
    p.add_argument("--min-trade-cash-pct", type=float, default=0.002, help="skip trades smaller than this fraction of equity")
    args = p.parse_args()

    ohlcv_map: Dict[str, List[Dict[str, Any]]] = {}
    symbols = args.symbols
    if args.universe:
        uni_syms = load_universe(args.universe)
        if uni_syms:
            symbols = uni_syms[: args.max_symbols]

    for s in symbols:
        rows = fetch_ohlcv(s, interval=args.interval, period=args.period)
        ohlcv_map[s] = rows
        # persist csv for re-use/inspection
        save_history_csv(history_path(s, args.interval, args.period), rows)

    # map profile to default rebalance cadence if not explicitly set
    default_reb = {"conservative": 21, "balanced": 10, "aggressive": 1}
    reb_n = args.rebalance_every if args.rebalance_every > 0 else default_reb.get(args.profile, 5)

    res = backtest(
        symbols,
        ohlcv_map,
        starting_cash=args.cash,
        profile=args.profile,
        max_holdings=args.max_holdings,
        rebalance_every=reb_n,
        fee_rate=args.fee_rate,
        fee_fixed=args.fee_fixed,
        slippage_bps=args.slippage_bps,
        min_trade_cash_pct=args.min_trade_cash_pct,
    )
    print(json.dumps({
        "symbols": symbols,
        "final_value": res["final_value"],
        "return_pct": (res["final_value"] - args.cash) / args.cash if args.cash else 0.0,
        "equity_points": len(res["equity_curve"]),
    }, indent=2))


if __name__ == "__main__":
    main()
