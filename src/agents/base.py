from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.types import TradingSignal


class Agent(ABC):
    """base agent, minimal api"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def signal(self, context: Dict[str, Any]) -> TradingSignal:
        raise NotImplementedError
