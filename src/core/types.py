from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SignalType = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class TradingSignal:
    symbol: str
    signal: SignalType
    confidence: float  # 0..1
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timestamp: Optional[float] = None  # epoch seconds


@dataclass(frozen=True)
class RiskProfile:
    name: Literal["conservative", "balanced", "aggressive"]
    target_return: str
    max_drawdown: float
    max_position_pct: float
    stop_loss_pct: float
    rebalance: Literal["monthly", "biweekly", "daily"]
