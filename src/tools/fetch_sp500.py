from __future__ import annotations

import os
from typing import List

from .universe import save_universe


def fetch_sp500() -> List[str]:
    try:
        import pandas as pd  # type: ignore
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = [str(x).strip() for x in df["Symbol"].tolist() if str(x).strip()]
        return syms
    except Exception:
        # minimal fallback small set
        return ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META"]


def main() -> None:
    syms = fetch_sp500()
    path = save_universe("sp500", syms)
    print(path)


if __name__ == "__main__":
    main()
