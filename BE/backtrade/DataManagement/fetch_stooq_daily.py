import pathlib
from typing import Iterable, List, Optional

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
    
    out = pathlib.Path(ROOT / "backtrade" / "DataManagement" / "data" / "stooq")
    #out.mkdir(parents=True, exist_ok=True)

    saved: List[pathlib.Path] = []
    for s in syms:
        # Use s, not "SPY"
        df = web.DataReader(s, "stooq")
        p = out / f"{s}.csv"
        df.to_csv(p)
        print(f"Saved {s}  →  {p}  ({len(df)} rows)")
        saved.append(p)

    return saved


if __name__ == "__main__":
    import sys

    # CLI usage: python fetch_stooq_daily.py AAPL MSFT SPY
    # If none provided, defaults kick in.
    import_stooq(sys.argv[1:] or None)
