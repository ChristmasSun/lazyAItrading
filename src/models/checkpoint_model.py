from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .torch_utils import try_import_torch


class MLPModel:
    """Small MLP for time-series features with checkpoint save/load."""

    def __init__(self, input_dim: int, hidden: int = 64):
        self.input_dim = input_dim
        self.hidden = hidden
        self.torch, self.nn, self.optim = try_import_torch()
        self.net = None
        if self.nn is not None:
            self.net = self.nn.Sequential(
                self.nn.Linear(input_dim, hidden),
                self.nn.ReLU(),
                self.nn.Linear(hidden, 1),
            )

    def is_available(self) -> bool:
        return self.net is not None and self.torch is not None

    def to(self, device: str = "cpu") -> None:
        if self.net is not None:
            self.net.to(device)

    def predict(self, X):
        if not self.is_available():
            # stub output
            return None
        with self.torch.no_grad():
            return self.net(X)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.is_available():
            # save torch weights to .pt alongside a json meta
            weights_path = path + ".pt"
            self.torch.save(self.net.state_dict(), weights_path)
            meta = {"input_dim": self.input_dim, "hidden": self.hidden, "weights": os.path.basename(weights_path)}
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        else:
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump({"message": "torch unavailable", "input_dim": self.input_dim, "hidden": self.hidden}, f)

    @classmethod
    def load(cls, path_no_ext: str) -> "MLPModel":
        # read meta json
        try:
            with open(path_no_ext + ".json", "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {"input_dim": 4, "hidden": 64}
        model = cls(input_dim=int(meta.get("input_dim", 4)), hidden=int(meta.get("hidden", 64)))
        if model.is_available():
            weights_path = path_no_ext + ".pt"
            if os.path.exists(weights_path):
                state = model.torch.load(weights_path, map_location="cpu")
                model.net.load_state_dict(state)
                model.net.eval()
        return model
