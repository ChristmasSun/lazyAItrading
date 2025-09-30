from __future__ import annotations

from src.agents.technical import TechnicalAnalysisAgent


def test_technical_agent_predictor_fallback():
    agent = TechnicalAnalysisAgent()
    # no artifact present by default; still should run
    sig = agent.signal({"symbol": "AAPL", "ohlcv": [{"close": 100, "high": 101, "low": 99, "open": 100, "volume": 1000}] * 60})
    assert sig.signal in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0