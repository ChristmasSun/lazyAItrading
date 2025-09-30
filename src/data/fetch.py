from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Optional
from typing import Any as _Any

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

from .cache import cache_path, is_valid, save, load


def fetch_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,  # e.g., "5y", "1y", "max"
) -> List[Dict[str, Any]]:
    """fetch OHLCV via yfinance, with disk cache and safe fallback"""
    if yf is None:
        from ..models.datasets import make_dummy_series

        return make_dummy_series(120)

    # defaults
    if end is None:
        end = dt.date.today().isoformat()
    if period is None and start is None:
        # default longer history if not specified
        start = (dt.date.today() - dt.timedelta(days=5 * 365)).isoformat()

    # cache
    # include period in cache key by extending start param representation
    cache_start = start if period is None else f"period:{period}"
    path = cache_path(symbol, cache_start, end, interval)
    if is_valid(path, ttl_seconds=6 * 60 * 60):  # 6h cache
        try:
            return load(path)
        except Exception:
            pass

    try:
        # explicit auto_adjust to avoid API default warnings
        # Use period if provided (mutually exclusive with start/end per yfinance)
        kwargs = {
            "interval": interval,
            "progress": False,
            "auto_adjust": False,
        }
        if period:
            kwargs.update({"period": period})
        else:
            kwargs.update({"start": start, "end": end})

        df = yf.download(symbol, **kwargs)
        df = df.dropna()
        if df is None or len(df) == 0:
            raise RuntimeError("empty df")

        # helper to extract scalar safely (handles possible single-element Series)
        def _to_float(val: _Any) -> float:
            try:
                # pandas Series single elem
                if hasattr(val, "iloc"):
                    return float(val.iloc[0])
                return float(val)
            except Exception:
                return 0.0
        out: List[Dict[str, Any]] = []
        for ts, row in df.iterrows():
            o = row["Open"] if "Open" in row else row.get(("Open", symbol), 0.0)
            h = row["High"] if "High" in row else row.get(("High", symbol), 0.0)
            l = row["Low"] if "Low" in row else row.get(("Low", symbol), 0.0)
            c = row["Close"] if "Close" in row else row.get(("Close", symbol), 0.0)
            v = row["Volume"] if "Volume" in row else row.get(("Volume", symbol), 0.0)

            out.append(
                {
                    "ts": int(ts.timestamp()),
                    "open": _to_float(o),
                    "high": _to_float(h),
                    "low": _to_float(l),
                    "close": _to_float(c),
                    "volume": _to_float(v),
                }
            )
        # write cache
        save(path, out)
        return out
    except Exception:
        from ..models.datasets import make_dummy_series

        return make_dummy_series(120)
