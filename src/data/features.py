from __future__ import annotations

from typing import Dict, Any, List

import math

try:
    import numpy as np  # type: ignore
except Exception:  # fallback minimal ops
    np = None  # type: ignore


def to_numpy(series: List[Dict[str, Any]]):
    if np is None:
        # simple python lists if numpy not installed
        close = [float(x.get("close", 0.0)) for x in series]
        high = [float(x.get("high", x.get("close", 0.0))) for x in series]
        low = [float(x.get("low", x.get("close", 0.0))) for x in series]
        volume = [float(x.get("volume", 0.0)) for x in series]
        return close, high, low, volume
    close = np.array([x.get("close", 0.0) for x in series], dtype=float)
    high = np.array([x.get("high", x.get("close", 0.0)) for x in series], dtype=float)
    low = np.array([x.get("low", x.get("close", 0.0)) for x in series], dtype=float)
    volume = np.array([x.get("volume", 0.0) for x in series], dtype=float)
    return close, high, low, volume


def rolling_mean(arr, window: int):
    if np is None:
        out = []
        for i in range(len(arr)):
            a = arr[max(0, i - window + 1) : i + 1]
            out.append(sum(a) / len(a))
        return out
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def pct_change(arr):
    if np is None:
        out = [0.0]
        for i in range(1, len(arr)):
            prev = arr[i - 1]
            out.append((arr[i] - prev) / prev if prev else 0.0)
        return out
    prev = np.roll(arr, 1)
    prev[0] = arr[0]
    return (arr - prev) / np.where(prev == 0, 1, prev)


def rolling_zscore(arr, window: int = 60):
    if np is None:
        out = []
        for i in range(len(arr)):
            a = arr[max(0, i - window + 1) : i + 1]
            mu = sum(a) / len(a)
            var = sum((x - mu) ** 2 for x in a) / len(a) if len(a) else 0.0
            sd = var ** 0.5
            out.append((arr[i] - mu) / sd if sd else 0.0)
        return out
    x = np.asarray(arr, dtype=float)
    # compute rolling mean and std via cumulative trick
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    out = np.zeros_like(x)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        n = i - start + 1
        s = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        s2 = csum2[i] - (csum2[start - 1] if start > 0 else 0.0)
        mu = s / n
        var = max(0.0, s2 / n - mu * mu)
        sd = np.sqrt(var)
        out[i] = (x[i] - mu) / sd if sd > 0 else 0.0
    return out


def build_features(series: List[Dict[str, Any]], lookback: int = 60) -> Dict[str, Any]:
    """return feature dict with arrays for model input"""
    if not series:
        return {"X": [], "last_price": 0.0}
    close, high, low, volume = to_numpy(series)
    ret = pct_change(close)
    vol = rolling_mean(volume, 20)
    # simple volatility proxy
    if np is None:
        volat = [0.0]
        for i in range(1, len(close)):
            volat.append(abs(close[i] - close[i - 1]))
    else:
        volat = np.abs(close - (np.roll(close, 1)))
        volat[0] = 0.0

    # stack last N
    N = min(lookback, len(close))
    feats = []
    for i in range(len(close) - N, len(close)):
        feats.append([
            ret[i],
            (high[i] - low[i]) / (close[i] if close[i] else 1.0),
            vol[i] / (volume[i] if volume[i] else 1.0),
            volat[i],
        ])
    return {"X": feats, "last_price": float(close[-1])}


def rsi(arr, period: int = 14):
    if np is None:
        gains, losses = [], []
        for i in range(1, len(arr)):
            diff = arr[i] - arr[i - 1]
            gains.append(max(0.0, diff))
            losses.append(max(0.0, -diff))
        ema_g = sum(gains[:period]) / period if period <= len(gains) else 0.0
        ema_l = sum(losses[:period]) / period if period <= len(losses) else 0.0
        out = [50.0] * len(arr)
        for i in range(period, len(arr)):
            ema_g = (ema_g * (period - 1) + gains[i - 1]) / period
            ema_l = (ema_l * (period - 1) + losses[i - 1]) / period
            rs = (ema_g / ema_l) if ema_l else 0.0
            out[i] = 100 - (100 / (1 + rs))
        return out
    diff = np.diff(arr, prepend=arr[0])
    gain = np.clip(diff, 0, None)
    loss = np.clip(-diff, 0, None)
    # simple EMA via convolution substitute
    def ema(x, n):
        alpha = 2 / (n + 1)
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    return 100 - (100 / (1 + rs))


def macd(arr, fast: int = 12, slow: int = 26, signal: int = 9):
    if np is None:
        return [0.0] * len(arr), [0.0] * len(arr)
    def ema(x, n):
        alpha = 2 / (n + 1)
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(high, low, close, period: int = 14):
    if np is None:
        tr = [high[0] - low[0]]
        for i in range(1, len(close)):
            tr_curr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr.append(tr_curr)
        out = []
        run = 0.0
        for i, v in enumerate(tr):
            run += v
            if i + 1 < period:
                out.append(run / (i + 1))
            else:
                run = run - tr[i - period + 1]
                out.append(run / period)
        return out
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # simple rolling mean as ATR approx
    return rolling_mean(tr, period)


def build_features_advanced(series: List[Dict[str, Any]], lookback: int = 120, normalize: bool = True) -> Dict[str, Any]:
    if not series:
        return {"X": [], "last_price": 0.0}
    close, high, low, volume = to_numpy(series)
    macd_line, macd_sig = macd(close)
    rsi_v = rsi(close)
    atr_v = atr(high, low, close)
    ret = pct_change(close)
    N = min(lookback, len(close))
    feats = []
    # optional normalization: z-score the continuous features for model stability
    if normalize:
        ret_n = rolling_zscore(ret, window=min(60, N))
        macd_delta = [float(macd_line[j] - macd_sig[j]) for j in range(len(close))]
        macd_n = rolling_zscore(macd_delta, window=min(60, N))
        rsi_n = [float((rsi_v[j] / 100.0 - 0.5) * 2.0) for j in range(len(close))]  # center RSI to [-1,1]
        atr_rel = [float(atr_v[j] / (close[j] if close[j] else 1.0)) for j in range(len(close))]
        atr_n = rolling_zscore(atr_rel, window=min(60, N))
        for i in range(len(close) - N, len(close)):
            feats.append([
                ret_n[i],
                macd_n[i],
                rsi_n[i],
                atr_n[i],
            ])
    else:
        for i in range(len(close) - N, len(close)):
            feats.append([
                ret[i],
                macd_line[i] - macd_sig[i],
                rsi_v[i] / 100.0,
                atr_v[i] / (close[i] if close[i] else 1.0),
            ])
    return {"X": feats, "last_price": float(close[-1])}
