from __future__ import annotations

import os
from typing import List

from .universe import save_universe


def fetch_sp500() -> List[str]:
    try:
        import pandas as pd  # type: ignore
        import requests  # type: ignore
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            raise RuntimeError("no tables parsed")
        df = tables[0]
        col = None
        # handle possible column name variants
        for c in ("Symbol", "Ticker symbol", "Ticker"):
            if c in df.columns:
                col = c
                break
        if col is None:
            raise RuntimeError("no symbol column in table")
        syms = []
        for x in df[col].tolist():
            s = str(x).strip().upper().replace(".", "-")  # BRK.B -> BRK-B style
            if s:
                syms.append(s)
        # dedupe and basic sanity filter
        syms = sorted(list(dict.fromkeys(syms)))
        # sometimes Wikipedia includes non-tickers; keep letters, numbers, and dashes
        syms = [s for s in syms if s and all(ch.isalnum() or ch == '-' for ch in s)]
        return syms
    except Exception:
        # minimal fallback small set
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META"]


def main() -> None:
    syms = fetch_sp500()
    path = save_universe("sp500", syms)
    print(f"{path} ({len(syms)} symbols)")


if __name__ == "__main__":
    main()
