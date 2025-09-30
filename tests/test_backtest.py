from __future__ import annotations

from src.backtest.engine import backtest


def test_backtest_smoke():
    # two symbols with synthetic upward trend
    series = [{"close": 100 + i, "high": 100 + i + 0.5, "low": 100 + i - 0.5, "open": 100 + i, "volume": 1000 + i} for i in range(120)]
    data = {"AAA": series, "BBB": series}
    res = backtest(["AAA", "BBB"], data, starting_cash=10_000.0, profile="balanced")
    assert res["final_value"] > 0
    assert len(res["equity_curve"]) > 0