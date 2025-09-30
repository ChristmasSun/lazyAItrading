from __future__ import annotations

import time
from typing import Any, Dict, List

from ..core.types import TradingSignal
from ..agents.base import Agent


class Orchestrator:
    """simple single-symbol flow for now"""

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def run_once(self, symbol: str, ohlcv: List[Dict[str, Any]], last_price: float) -> TradingSignal:
        ctx: Dict[str, Any] = {"symbol": symbol, "ohlcv": ohlcv, "last_price": last_price}

        # technical first
        tech = next(a for a in self.agents if a.name == "technical")
        tech_sig = tech.signal(ctx)

        # pass through risk
        risk = next(a for a in self.agents if a.name == "risk")
        final_sig = risk.signal({"base_signal": tech_sig, **ctx})

        return TradingSignal(
            symbol=final_sig.symbol,
            signal=final_sig.signal,
            confidence=final_sig.confidence,
            price_target=final_sig.price_target,
            stop_loss=final_sig.stop_loss,
            reasoning=final_sig.reasoning,
            timestamp=time.time(),
        )
