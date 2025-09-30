from __future__ import annotations

import json
from typing import Any, Dict

from .core.orchestrator import Orchestrator
from .agents.technical import TechnicalAnalysisAgent
from .agents.risk import RiskAgent
from .models.datasets import make_dummy_series


def run_once(symbol: str = "AAPL", profile: str = "balanced") -> Dict[str, Any]:
    orch = Orchestrator([TechnicalAnalysisAgent(), RiskAgent(profile)])
    ohlcv = make_dummy_series(60)
    last_price = ohlcv[-1]["close"] if ohlcv else 100.0
    sig = orch.run_once(symbol, ohlcv, last_price)
    return {
        "symbol": sig.symbol,
        "signal": sig.signal,
        "confidence": sig.confidence,
        "stop_loss": sig.stop_loss,
        "reasoning": sig.reasoning,
    }


if __name__ == "__main__":
    out = run_once()
    print(json.dumps(out, indent=2))
