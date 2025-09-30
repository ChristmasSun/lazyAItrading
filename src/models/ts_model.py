from __future__ import annotations

from typing import Any, Optional
import importlib


class TinyTSModel:
    """basic time-series model skeleton. uses torch if available"""

    def __init__(self, input_dim: int = 4, hidden: int = 32):
        self.input_dim = input_dim
        self.hidden = hidden
        self._torch = None
        self._net = None
        try:
            torch = importlib.import_module("torch")
            nn = importlib.import_module("torch.nn")
            self._torch = torch  # type: ignore[assignment]
            self._net = nn.Sequential(  # type: ignore[attr-defined]
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        except Exception:
            # torch not installed; keep stub
            pass

    def forward(self, x: Any) -> Any:
        if self._net is None:
            # stub output
            return 0.0
        return self._net(x)
