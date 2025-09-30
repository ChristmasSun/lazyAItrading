from __future__ import annotations

from typing import Any, List
import importlib


class TSDataset:
    """windowed time-series dataset; works without torch (stub)"""

    def __init__(self, X: List[List[float]], y: List[float]):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_dataloader(X: List[List[float]], y: List[float], batch_size: int = 32):
    """return torch DataLoader if available, else python iterator stub"""
    try:
        torch = importlib.import_module("torch")
        data = importlib.import_module("torch.utils.data")
        import numpy as np  # type: ignore

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)
        ds = data.TensorDataset(X_t, y_t)
        return data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    except Exception:
        # simple generator fallback
        def gen():
            for i in range(0, len(X), batch_size):
                yield X[i : i + batch_size], y[i : i + batch_size]

        return gen()
