from __future__ import annotations

from typing import Dict, Any, List

from ..backtest.portfolio import Portfolio
from ..agents.technical import TechnicalAnalysisAgent
from ..agents.risk import RiskAgent
from ..selection.selector import score_universe


def backtest(
    symbols: List[str],
    ohlcv_map: Dict[str, List[Dict[str, Any]]],
    starting_cash: float = 10_000.0,
    profile: str = "balanced",
    max_holdings: int = 10,
    rebalance_every: int = 5,
    fee_rate: float = 0.0005,
    fee_fixed: float = 0.0,
    slippage_bps: float = 2.0,
    min_trade_cash_pct: float = 0.002
) -> Dict[str, Any]:
    port = Portfolio(cash=starting_cash, fee_rate=fee_rate, fee_fixed=fee_fixed, slippage_bps=slippage_bps)
    tech = TechnicalAnalysisAgent()
    risk = RiskAgent(profile)

    # assume all series aligned to the same length by last N
    lengths = [len(ohlcv_map[s]) for s in symbols if s in ohlcv_map]
    if not lengths:
        return {"equity_curve": [], "final_value": starting_cash}
    N = min(lengths)

    equity_curve = []
    for i in range(N):
        # step prices for marking
        prices = {}
        for sym in symbols:
            series = ohlcv_map.get(sym, [])
            bar = series[-N + i]
            prices[sym] = float(bar.get("close", 0.0))

        # periodic selection and rebalance
        if rebalance_every > 0 and i % rebalance_every == 0:
            ranked = score_universe(symbols, ohlcv_map)
            picks = [s for s, _ in ranked[:max_holdings]]
            # target equal-weight across picks capped by risk profile per position
            eq = port.value(prices)
            per_pos_cash = min(eq * risk.profile.max_position_pct, eq / max(1, len(picks)))

            # sell positions not in picks
            for sym, pos in list(port.positions.items()):
                if pos.qty > 0 and sym not in picks:
                    port.sell(sym, prices.get(sym, 0.0), pos.qty)

            # buy/adjust picks
            for sym in picks:
                px = prices.get(sym, 0.0)
                if px <= 0:
                    continue
                pos = port.positions.get(sym)
                current_val = (pos.qty * px) if pos else 0.0
                delta_cash = max(0.0, per_pos_cash - current_val)
                # skip tiny trades to reduce churn
                if delta_cash > max(min_trade_cash_pct * eq, 1.0):
                    port.buy(sym, px, delta_cash)

        # enforce stop-loss daily
        for sym, pos in list(port.positions.items()):
            if pos.qty > 0:
                px = prices.get(sym, 0.0)
                # build a synthetic signal with current price to compute stop
                final_stop = risk.profile.stop_loss_pct
                stop_px = px * (1 - final_stop)
                if px <= stop_px:
                    port.sell(sym, px, pos.qty)

        equity_curve.append(port.value(prices))

    return {"equity_curve": equity_curve, "final_value": equity_curve[-1] if equity_curve else starting_cash}
