#!/usr/bin/env python3
"""
av_downloader.py — Alpha Vantage→Backtrader CSV fetcher (free‑tier safe)
============================================================================
*One stop for daily **or** intraday bars that never trips the “premium
endpoint” wall.*  The script:

1. Reads all symbols from your `config.yaml` (both `symbols:` or
   `runs:` layouts).
2. **Daily bars**: tries `TIME_SERIES_DAILY_ADJUSTED` *first*; if AV
   replies with the “premium endpoint” JSON, it transparently falls back
   to the free‑tier `TIME_SERIES_DAILY` endpoint.
3. **Intraday bars** (1/5/15/30/60 min) call `TIME_SERIES_INTRADAY`.
4. Re‑orders to *ascending* timestamps and keeps only the six columns
   Backtrader expects: `Date,Open,High,Low,Close,Volume`.
5. Sleeps 12 s between calls so you’ll never exceed the 5‑per‑minute
   limit.

Usage:
------
```bash
export AV_KEY=YOUR_FREE_API_KEY   # one‑time (env var preferred)
python av_downloader.py           # uses config.yaml → DataManagement/data

# pick 5‑minute bars instead of daily
python av_downloader.py --intraday 5
```

Tip: schedule this with cron / Task Scheduler at 01:00 local and your
CSV lake grows automatically each night.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
import requests
import yaml
import pathlib
from typing import Iterable, List, Optional
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AV_BASE = "https://www.alphavantage.co/query?datatype=csv&outputsize=compact"


def parse_yaml_symbols(cfg_path: Path | str) -> list[str]:
    """Return a de‑duplicated, order‑preserving list of symbols from YAML."""
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if "symbols" in cfg:
        raw: Iterable[str | dict] = cfg["symbols"]
        symbols = [s if isinstance(s, str) else s.get("symbol") for s in raw]
    else:  # fallback: runs: - symbol: SPY
        symbols = [run["symbol"] for run in cfg.get("runs", [])]

    seen: set[str] = set()
    uniq: list[str] = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# ---------- Alpha Vantage wrappers ----------------------------------------


def _av_get(params: str, api_key: str) -> str:
    print("[INFO] avdownloader._av_get : used following api_key",api_key)
    url = f"{AV_BASE}&apikey={api_key}&{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_daily_csv(symbol: str, api_key: str) -> tuple[str, str]:
    """Return (csv_text, endpoint_used) for daily bars with auto‑fallback."""
    # 1) try the fully‑adjusted endpoint (may be premium for big tickers)
    txt = _av_get(f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}", api_key)
    if _looks_like_premium(txt):
        time.sleep(2)
        # 2) fall back to the free un‑adjusted endpoint
        txt = _av_get(f"function=TIME_SERIES_DAILY&symbol={symbol}", api_key)
        if _looks_like_premium(txt):
            raise RuntimeError(txt.strip())  # both failed → propagate
        return txt, "TIME_SERIES_DAILY"
    return txt, "TIME_SERIES_DAILY_ADJUSTED"


def fetch_intraday_csv(symbol: str, interval: int, api_key: str) -> str:
    return _av_get(
        f"function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}min",
        api_key,
    )


def _looks_like_premium(text: str) -> bool:
    return text.startswith("{\"Information\"") or ",open," not in text


# ---------- CSV massaging --------------------------------------------------


def convert_to_bt_rows(raw_csv: str) -> list[list[str]]:
    lines = raw_csv.splitlines()
    if not lines:
        raise RuntimeError("Empty response from Alpha Vantage")

    header = lines[0].strip()
    reader = csv.DictReader(lines)

    # Accept a few common variants just in case
    ts_key = None
    for k in ("timestamp", "date", "time"):
        if k in (reader.fieldnames or []):
            ts_key = k
            break

    if not ts_key:
        # Show the header (and a tiny snippet) so you immediately see what AV returned
        snippet = "\n".join(lines[:3])
        raise RuntimeError(f"Unexpected CSV header (no timestamp): {header}\nFirst lines:\n{snippet}")

    rows = []
    for row in reader:
        rows.append([
            row[ts_key][:19],
            row.get("open", ""),
            row.get("high", ""),
            row.get("low", ""),
            row.get("close", ""),
            row.get("volume", ""),
        ])

    rows.sort(key=lambda r: r[0])
    return rows



def save_rows(path: Path, rows: list[list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def av_doawnloader_main(configFile: str):


    from pathlib import Path
    ROOT = Path(__file__).parent

    out = Path(ROOT / "data" )#ROOT here is relative to the current file path



    ap = argparse.ArgumentParser(description="Download Alpha Vantage CSVs for Backtrader")
    ap.add_argument("-c", "--config", default=configFile, help="YAML config (default: config.yaml)")
    ap.add_argument("-o", "--outdir", default=out, help="Output dir (default: ROOT/backtrade /DataManagement/data)")
    ap.add_argument("--intraday", type=int, choices=[1, 5, 15, 30, 60], help="Interval in minutes (skip for daily)")
    opts = ap.parse_args(args=[])

    api_key = os.getenv("AV_KEY") or os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key: sys.exit("Set AV_KEY environment variable with your Alpha Vantage API key.")

    symbols = parse_yaml_symbols(opts.config)
    if not symbols: sys.exit("No symbols found in YAML.")

    outdir = Path(opts.outdir)
    intrv = opts.intraday

    print(f"[INFO] av_doawnloader.av_doawnloader_main : attempting to fetch {symbols} symbol(s) from Alpha Vantage …")

    for idx, sym in enumerate(symbols, 1):
        try:
            if intrv:
                raw = fetch_intraday_csv(sym, intrv, api_key)
                fname = outdir / f"{sym}_{intrv}m.csv"
                endpoint = f"INTRADAY {intrv}m"
            else:
                raw, endpoint = fetch_daily_csv(sym, api_key)
                suffix = "_adj" if endpoint.endswith("ADJUSTED") else ""
                fname = outdir / f"{sym}{suffix}.csv"

            rows = convert_to_bt_rows(raw)
            save_rows(fname, rows)
            print(f"[{idx}/{len(symbols)}] {sym} ← {endpoint}  → {fname}  ({len(rows)} rows)")

        except Exception as exc:
            print(f"[Exception] av_doawnloader.av_doawnloader_main : {sym}: {exc}")

        # Free tier: 5 calls / min (12‑second gap keeps us safe)
        if idx < len(symbols):
            time.sleep(12)

    print("[INFO] av_doawnloader.av_doawnloader_main : Done.  You can now point backtrader_yaml_runner at the new CSVs.")






def import_stooq(symbols: Optional[Iterable[str]] = None) -> List[pathlib.Path]:
    """

    - If `symbols` is None: uses a small default list (handy for CLI/tests).
    - IMPORTANT: does NOT read sys.argv (so it is safe to call from a running API server).
    """
    syms = list(symbols)
    from pathlib import Path
    ROOT = Path(__file__).parent
    
    out = Path(ROOT / "data")#ROOT here is relative to the current file path
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
    out = ROOT / "data"    
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
