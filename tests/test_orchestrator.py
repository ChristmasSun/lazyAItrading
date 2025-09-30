from __future__ import annotations

from src.core.orchestrator import Orchestrator
from src.agents.technical import TechnicalAnalysisAgent
from src.agents.risk import RiskAgent


def test_run_once_balanced_profile():
    orch = Orchestrator([TechnicalAnalysisAgent(), RiskAgent("balanced")])
    sig = orch.run_once("AAPL", ohlcv=[{"close": 100}], last_price=100.0)
    assert sig.symbol == "AAPL"
    assert sig.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.stop_loss is not None
