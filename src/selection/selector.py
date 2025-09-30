from __future__ import annotations

from typing import Dict, Any, List, Tuple

from ..agents.technical import TechnicalAnalysisAgent


def score_universe(symbols: List[str], ohlcv_map: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, float]]:
    agent = TechnicalAnalysisAgent()
    scores: List[Tuple[str, float]] = []
    for s in symbols:
        series = ohlcv_map.get(s, [])
        ctx = {"symbol": s, "ohlcv": series[-60:], "last_price": (series[-1]["close"] if series else 0.0)}
        sig = agent.signal(ctx)
        val = sig.confidence if sig.signal != "HOLD" else sig.confidence * 0.5
        # tie break by favoring BUY slightly
        if sig.signal == "BUY":
            val += 0.05
        scores.append((s, float(val)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
