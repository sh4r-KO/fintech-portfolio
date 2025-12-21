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




def import_yahoo(symbols: Iterable[str], start, end) -> List[pathlib.Path]:
    if not symbols:
        return []

    syms = list(symbols)
    ROOT = Path(__file__).parent
    out = ROOT / "data" / "stooq"   # (you may want "data/yahoo" instead)
    out.mkdir(parents=True, exist_ok=True)

    saved: List[pathlib.Path] = []

    for s in syms:
        try:
            df = yf.download(
                s,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",   # helps keep columns as OHLCV
            )

            if df is None or df.empty:
                print(f"[WARN] Yahoo returned no data for {s}")
                continue

            # ---- FIX 1: flatten MultiIndex columns if present ----
            if isinstance(df.columns, pd.MultiIndex):
                # For yfinance, the first level is usually 'Price'/'Open' etc.
                # We want the OHLCV names only.
                df.columns = df.columns.get_level_values(0)

            # ---- FIX 2: normalize exactly to Backtrader CSV shape ----
            df = df.rename(columns=str.title)
            df = df.sort_index()
            df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) else df.index

            # Keep only OHLCV (and ensure they exist)
            need = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in need if c not in df.columns]
            if missing:
                print(f"[ERR] Missing columns for {s}: {missing}. Got: {list(df.columns)}")
                continue

            out_df = df[need].copy()
            out_df.index.name = "Date"

            # ---- FIX 3: write Date as first column, no MultiIndex header rows ----
            p = out / f"{s}.csv"
            out_df.reset_index().to_csv(p, index=False)

            print(f"Saved {s}  →  {p}  ({len(out_df)} rows)")
            saved.append(p)

        except Exception as e:
            print(f"[ERR] Yahoo download failed for {s}: {e}")

    return saved


if __name__ == "__main__":
    import sys

    # CLI usage: python fetch_stooq_daily.py AAPL MSFT SPY
    # If none provided, defaults kick in.
    import_stooq(sys.argv[1:] or None)
