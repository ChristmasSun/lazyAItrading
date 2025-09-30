from __future__ import annotations

from src.data.fetch import fetch_ohlcv
from src.data.features import build_features


def test_build_features_dummy_ok():
    data = fetch_ohlcv("AAPL")  # will fallback to dummy if yfinance missing
    feats = build_features(data, lookback=30)
    assert "X" in feats and isinstance(feats["X"], list)
    assert len(feats["X"]) > 0
    row = feats["X"][0]
    assert isinstance(row, list) and len(row) == 4