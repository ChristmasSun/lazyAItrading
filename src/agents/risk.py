from __future__ import annotations

from typing import Any, Dict

from .base import Agent
from ..core.types import TradingSignal, RiskProfile


class RiskAgent(Agent):
    """enforce sizing and stops per profile"""

    PROFILES: Dict[str, RiskProfile] = {
        "conservative": RiskProfile(
            name="conservative",
            target_return="8-12%",
            max_drawdown=0.05,
            max_position_pct=0.02,
            stop_loss_pct=0.03,
            rebalance="monthly",
        ),
        "balanced": RiskProfile(
            name="balanced",
            target_return="15-25%",
            max_drawdown=0.12,
            max_position_pct=0.05,
            stop_loss_pct=0.07,
            rebalance="biweekly",
        ),
        "aggressive": RiskProfile(
            name="aggressive",
            target_return="30%+",
            max_drawdown=0.20,
            max_position_pct=0.08,
            stop_loss_pct=0.15,
            rebalance="daily",
        ),
    }

    def __init__(self, profile: str = "balanced"):
        super().__init__(name="risk")
        if profile not in self.PROFILES:
            raise ValueError("bad profile")
        self.profile = self.PROFILES[profile]

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # maybe calc exposure etc later
        return {"profile": self.profile.name}

    def apply(self, sig: TradingSignal, price: float) -> TradingSignal:
        # apply stop loss, keep target optional
        stop = price * (1 - self.profile.stop_loss_pct) if price else None
        return TradingSignal(
            symbol=sig.symbol,
            signal=sig.signal,
            confidence=min(1.0, sig.confidence),
            price_target=sig.price_target,
            stop_loss=stop,
            reasoning=(sig.reasoning + f" | risk:{self.profile.name}").strip(),
            timestamp=sig.timestamp,
        )

    def signal(self, context: Dict[str, Any]) -> TradingSignal:
        base: TradingSignal = context["base_signal"]
        price = context.get("last_price", 0.0)
        return self.apply(base, price)
