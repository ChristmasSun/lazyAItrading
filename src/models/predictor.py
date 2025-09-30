from __future__ import annotations

import json
import os
from typing import Dict, Any, List


class Predictor:
    def __init__(self, artifact: str | None = None):
        self.artifact = artifact
        self.meta: Dict[str, Any] | None = None
        if artifact and os.path.exists(artifact):
            try:
                with open(artifact, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = None

    def predict(self, X: List[List[float]]) -> Dict[str, Any]:
        # if we had torch and real weights, we would load and run the model here
        # for now, use a simple rule: average the last feature
        if not X:
            return {"signal": "HOLD", "confidence": 0.5, "price_change": 0.0}
        last_row = X[-1]
        score = float(sum(last_row) / len(last_row)) if last_row else 0.0
        signal = "BUY" if score > 0 else ("SELL" if score < 0 else "HOLD")
        confidence = min(1.0, abs(score))
        return {"signal": signal, "confidence": confidence, "price_change": score}
