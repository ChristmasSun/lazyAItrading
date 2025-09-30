from __future__ import annotations

from typing import Any, Dict

from .base import Agent
from ..core.types import TradingSignal
from ..models.market_dynamics import MarketDynamicsModel
from ..models.predictor import Predictor
from ..models.checkpoint_model import MLPModel
from ..data.features import build_features, build_features_advanced


class TechnicalAnalysisAgent(Agent):
    """stub tech agent. later swap in custom models"""

    def __init__(self):
        super().__init__(name="technical")
        self.model = MarketDynamicsModel()
        # try to use trained checkpoint (JSON + .pt) else predictor
        self.checkpoint_path = "artifacts/models/tiny"
        self.ckpt_model = MLPModel.load(self.checkpoint_path)
        self.predictor = Predictor(artifact=self.checkpoint_path + ".json")

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context.get("ohlcv", [])
        # build advanced features if possible else basic
        feats = build_features_advanced(data, lookback=60)
        if not feats["X"]:
            feats = build_features(data, lookback=60)
        # if torch model loaded, prefer it else predictor then fallback
        pred: Dict[str, Any]
        if getattr(self.ckpt_model, "is_available")() and self.ckpt_model.net is not None and feats["X"]:
            # simple torch forward: compute average score
            try:
                import numpy as np  # type: ignore
                import importlib

                torch = importlib.import_module("torch")
                X_t = torch.tensor(np.asarray(feats["X"]), dtype=torch.float32)
                out = self.ckpt_model.predict(X_t)
                # treat output as next-step return estimate
                score = float(out[-1].item()) if out is not None else 0.0
                pred = {"price_change": score, "confidence": min(1.0, abs(score)), "signal": "BUY" if score > 0 else ("SELL" if score < 0 else "HOLD")}
            except Exception:
                pred = self.predictor.predict(feats["X"]) if feats["X"] else self.model.predict([])
        else:
            pred = self.predictor.predict(feats["X"]) if feats["X"] else self.model.predict([])
        strength = 0.5 + float(pred.get("price_change", 0.0))
        # bound strength to [0,1] to keep semantics stable
        strength = max(0.0, min(1.0, strength))
        conf = float(pred.get("confidence", 0.5))
        return {"strength": strength, "confidence": conf}

    def signal(self, context: Dict[str, Any]) -> TradingSignal:
        info = self.analyze(context)
        strength = info["strength"]
        action = "BUY" if strength > 0.55 else ("SELL" if strength < 0.45 else "HOLD")
        # derive confidence and clamp to [0,1]
        conf_est = max(info.get("confidence", 0.5), abs(strength - 0.5) * 2)
        conf_est = max(0.0, min(1.0, conf_est))
        return TradingSignal(
            symbol=context.get("symbol", "UNK"),
            signal=action,  # type: ignore[arg-type]
            confidence=conf_est,
            reasoning="tech+mdm",
        )
