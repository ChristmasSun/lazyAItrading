from __future__ import annotations

import os
from typing import List, Tuple
import importlib

from .data.fetch import fetch_ohlcv
from .data.features import build_features_advanced, to_numpy, pct_change
from .models.checkpoint_model import MLPModel
from .models.torch_utils import try_import_torch


def train_torch(symbols: List[str], out_no_ext: str, epochs: int = 10, batch_size: int = 128, max_days: int = 365, val_frac: float = 0.2, patience: int = 3) -> str:
    # collect features (concat across symbols)
    X: List[List[float]] = []
    y: List[float] = []
    for sym in symbols:
        series = fetch_ohlcv(sym)
        # limit data size by lookback window (roughly max_days points with daily interval)
        series = series[-max_days:]
        feats = build_features_advanced(series, lookback=min(240, len(series)), normalize=True)
        X_sym = feats["X"]
        # derive next-step return as label
        closes = to_numpy(series)[0]
        rets = pct_change(closes)
        # align: label for row i is ret[i+1]; drop last if no next
        y_sym = rets[-len(X_sym) - 1 : -1] if len(rets) >= len(X_sym) + 1 else [0.0] * len(X_sym)
        X.extend(X_sym)
        y.extend(y_sym)

    torch, nn, optim = try_import_torch()
    if torch is None or nn is None or optim is None or not X:
        # save stub meta if torch unavailable
        model = MLPModel(input_dim=len(X[0]) if X else 4)
        model.save(out_no_ext)
        return out_no_ext

    import numpy as np  # type: ignore

    X_all = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_all = torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)

    # train/val split
    n = X_all.shape[0]
    n_val = max(1, int(n * val_frac))
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    y_train, y_val = y_all[:-n_val], y_all[-n_val:]

    model = MLPModel(input_dim=X_all.shape[1])
    model.to("cpu")

    assert model.net is not None
    optimizer = optim.Adam(model.net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    bad_epochs = 0
    m = X_train.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(m)
        X_train = X_train[perm]
        y_train = y_train[perm]
        # train
        model.net.train()  # type: ignore[union-attr]
        for i in range(0, m, batch_size):
            xb = X_train[i : i + batch_size]
            yb = y_train[i : i + batch_size]
            optimizer.zero_grad()
            out = model.net(xb)  # type: ignore[arg-type]
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
        # val
        model.net.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            val_out = model.net(X_val)  # type: ignore[arg-type]
            val_loss = float(loss_fn(val_out, y_val).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            # save best checkpoint immediately
            model.save(out_no_ext)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    # ensure at least one checkpoint exists
    if best_val == float("inf"):
        model.save(out_no_ext)
    return out_no_ext
