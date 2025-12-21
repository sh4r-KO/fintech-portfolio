import pathlib
from typing import Iterable, List, Optional
from pathlib import Path

import yfinance as yf
import pandas_datareader.data as web


def import_stooq(symbols: Optional[Iterable[str]] = None) -> List[pathlib.Path]:
    """
    Download daily history from Stooq for each symbol and save to DataManagement/data/stooq/<SYMBOL>_d.csv.

    - If `symbols` is None: uses a small default list (handy for CLI/tests).
    - IMPORTANT: does NOT read sys.argv (so it is safe to call from a running API server).
    """
    syms = list(symbols)
    from pathlib import Path
    ROOT = Path(__file__).parent
    
    out = Path(ROOT / "data" / "stooq")#ROOT here is relative to the current file path
    #out.mkdir(parents=True, exist_ok=True)

    saved: List[pathlib.Path] = []
    for s in syms:
        # Use s, not "SPY"
        df = web.DataReader(s, "stooq")
        df = df.sort_index()  # <-- oldest -> newest

        p = out / f"{s}.csv"
        df.to_csv(p)
        print(f"Saved {s}  →  {p}  ({len(df)} rows)")
        saved.append(p)

    return saved




def import_yahoo(symbols: Optional[Iterable[str]] = None) -> List[pathlib.Path]:
    """
    Download daily history from Yahoo Finance for each symbol and save to
    data/yahoo/<SYMBOL>.csv.

    - Uses yfinance (no API key required)
    - Downloads full available daily history
    - Sorts oldest -> newest (Backtrader-friendly)
    - Safe to call from an API server (no argv usage)
    """
    if not symbols:
        return []

    syms = list(symbols)

    ROOT = Path(__file__).parent
    out = ROOT / "data" / "yahoo"
    out.mkdir(parents=True, exist_ok=True)

    saved: List[pathlib.Path] = []

    for s in syms:
        try:
            df = yf.download(
                s,
                period="max",        # full history Yahoo has
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df.empty:
                print(f"[WARN] Yahoo returned no data for {s}")
                continue

            # Normalize column names to Backtrader expectations
            df = df.rename(columns=str.title)

            # Ensure DatetimeIndex, sorted ascending
            df.index.name = "Date"
            df = df.sort_index()

            # Keep only OHLCV (Yahoo sometimes adds extras)
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            p = out / f"{s}.csv"
            df.to_csv(p)

            print(f"Saved {s}  →  {p}  ({len(df)} rows)")
            saved.append(p)

        except Exception as e:
            print(f"[ERR] Yahoo download failed for {s}: {e}")

    return saved



if __name__ == "__main__":
    import sys

    # CLI usage: python fetch_stooq_daily.py AAPL MSFT SPY
    # If none provided, defaults kick in.
    import_stooq(sys.argv[1:] or None)
