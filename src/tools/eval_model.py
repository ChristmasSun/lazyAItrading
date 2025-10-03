from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

from ..data.fetch import fetch_ohlcv
from ..backtest.engine import backtest
from ..backtest.portfolio import Portfolio


def compute_metrics(equity_curve: List[float], starting_cash: float = 10_000.0) -> Dict[str, Any]:
    """compute sharpe, drawdown, returns from equity curve"""
    if not equity_curve:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "final_equity": starting_cash,
        }
    
    # returns
    final = equity_curve[-1]
    total_ret = (final - starting_cash) / starting_cash
    
    # daily returns for sharpe
    rets = []
    for i in range(1, len(equity_curve)):
        r = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1] if equity_curve[i - 1] > 0 else 0.0
        rets.append(r)
    
    mean_ret = sum(rets) / len(rets) if rets else 0.0
    var = sum((r - mean_ret) ** 2 for r in rets) / len(rets) if rets else 0.0
    std = var ** 0.5
    # annualized sharpe (assuming ~252 trading days)
    sharpe = (mean_ret / std * (252 ** 0.5)) if std > 0 else 0.0
    
    # max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    
    # win rate (% of positive return days)
    wins = sum(1 for r in rets if r > 0)
    win_rate = wins / len(rets) if rets else 0.0
    
    return {
        "total_return": total_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "final_equity": final,
        "num_periods": len(equity_curve),
    }


def baseline_buy_hold(symbols: List[str], ohlcv_map: Dict[str, List[Dict[str, Any]]], cash: float) -> List[float]:
    """equal-weight buy and hold baseline"""
    # buy equal weight at start, hold til end
    port = Portfolio(cash=cash)
    
    # get min length
    lengths = [len(ohlcv_map[s]) for s in symbols if s in ohlcv_map]
    if not lengths:
        return [cash]
    N = min(lengths)
    
    # buy at bar 0
    prices_0 = {}
    for s in symbols:
        if s in ohlcv_map:
            prices_0[s] = float(ohlcv_map[s][-N]["close"])
    
    alloc = cash / len(symbols) if symbols else 0.0
    for s in symbols:
        px = prices_0.get(s, 0.0)
        if px > 0:
            port.buy(s, px, alloc)
    
    # mark to market each bar
    curve = []
    for i in range(N):
        prices = {}
        for s in symbols:
            if s in ohlcv_map:
                prices[s] = float(ohlcv_map[s][-N + i]["close"])
        curve.append(port.value(prices))
    
    return curve


def eval_model(
    symbols: List[str],
    cash: float = 10_000.0,
    profile: str = "balanced",
    max_holdings: int = 10,
    interval: str = "1d",
    period: str = "1y",
) -> Dict[str, Any]:
    """backtest model vs baseline and compute metrics"""
    
    # fetch data
    ohlcv_map = {}
    for s in symbols:
        series = fetch_ohlcv(s, interval=interval, period=period)
        if series:
            ohlcv_map[s] = series
    
    if not ohlcv_map:
        return {"error": "no_data"}
    
    # run model backtest
    model_result = backtest(
        symbols=symbols,
        ohlcv_map=ohlcv_map,
        starting_cash=cash,
        profile=profile,
        max_holdings=max_holdings,
        rebalance_every=5,
    )
    
    # run baseline
    baseline_curve = baseline_buy_hold(symbols, ohlcv_map, cash)
    
    # compute metrics
    model_metrics = compute_metrics(model_result["equity_curve"], cash)
    baseline_metrics = compute_metrics(baseline_curve, cash)
    
    # alpha = model return - baseline return
    alpha = model_metrics["total_return"] - baseline_metrics["total_return"]
    
    return {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "alpha": alpha,
        "symbols": symbols,
        "profile": profile,
        "num_symbols": len(symbols),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate model performance vs baseline")
    ap.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"])
    ap.add_argument("--cash", type=float, default=10_000.0)
    ap.add_argument("--profile", type=str, default="balanced")
    ap.add_argument("--max-holdings", type=int, default=10)
    ap.add_argument("--interval", type=str, default="1d")
    ap.add_argument("--period", type=str, default="1y")
    ap.add_argument("--out", type=str, default="artifacts/models/eval.json", help="output JSON path")
    args = ap.parse_args()
    
    print(f"evaluating model on {len(args.symbols)} symbols...")
    result = eval_model(
        symbols=args.symbols,
        cash=args.cash,
        profile=args.profile,
        max_holdings=args.max_holdings,
        interval=args.interval,
        period=args.period,
    )
    
    # save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    # print summary
    print("\n=== Model Performance ===")
    print(f"Total Return:  {result['model']['total_return']*100:>7.2f}%")
    print(f"Sharpe Ratio:  {result['model']['sharpe_ratio']:>7.2f}")
    print(f"Max Drawdown:  {result['model']['max_drawdown']*100:>7.2f}%")
    print(f"Win Rate:      {result['model']['win_rate']*100:>7.2f}%")
    print(f"Final Equity:  ${result['model']['final_equity']:>7.2f}")
    
    print("\n=== Baseline (Buy & Hold) ===")
    print(f"Total Return:  {result['baseline']['total_return']*100:>7.2f}%")
    print(f"Sharpe Ratio:  {result['baseline']['sharpe_ratio']:>7.2f}")
    print(f"Max Drawdown:  {result['baseline']['max_drawdown']*100:>7.2f}%")
    print(f"Win Rate:      {result['baseline']['win_rate']*100:>7.2f}%")
    print(f"Final Equity:  ${result['baseline']['final_equity']:>7.2f}")
    
    print(f"\n=== Alpha ===")
    print(f"Excess Return: {result['alpha']*100:>7.2f}%")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
