from fastapi import FastAPI, HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr,Field
from typing import List, Optional
from pathlib import Path
import json

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import importlib
import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "projects.json"

app = FastAPI(title="Fintech Portfolio API", version="1.0.0")


class Link(BaseModel):
    github: Optional[str] = None
    demo: Optional[str] = None
    paper: Optional[str] = None

class Metrics(BaseModel):
    rows_processed: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_savings_pct: Optional[float] = None

class Project(BaseModel):
    id: str; slug: str; title: str; summary: str
    tags: List[str] = []; tech: List[str] = []
    problem: Optional[str] = None; approach: Optional[str] = None
    results: Optional[str] = None; cover_image: Optional[str] = None
    links: Optional[Link] = None; metrics: Optional[Metrics] = None

class Contact(BaseModel):
    name: str; email: EmailStr; message: str

PROJECTS: List[Project] = []
INDEX_BY_SLUG = {}

def refresh_data():
    global PROJECTS, INDEX_BY_SLUG
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            PROJECTS = [Project(**p) for p in json.load(f)]
        INDEX_BY_SLUG = {p.slug: p for p in PROJECTS}
        print(f"[startup] Loaded {len(PROJECTS)} projects")
    except FileNotFoundError:
        print(f"[startup] WARNING: {DATA_PATH} missing; starting empty")
        PROJECTS, INDEX_BY_SLUG = [], {}
    except Exception as e:
        print(f"[startup] ERROR loading data: {e}")
        PROJECTS, INDEX_BY_SLUG = [], {}

@app.on_event("startup")
def on_startup():
    refresh_data()

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/projects", response_model=List[Project])
def list_projects(q: Optional[str] = None, tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    items = PROJECTS
    if q:
        ql = q.lower()
        items = [p for p in items if ql in p.title.lower() or ql in p.summary.lower()
                 or any(ql in t.lower() for t in p.tech + p.tags)]
    if tag:
        tl = tag.lower()
        items = [p for p in items if any(tl == t.lower() for t in p.tags)]
    return items[offset: offset + limit]

@app.get("/api/projects/{slug}", response_model=Project)
def get_project(slug: str):
    p = INDEX_BY_SLUG.get(slug)
    if not p: raise HTTPException(status_code=404, detail="Project not found")
    return p

@app.post("/api/contact")
def contact(payload: Contact):
    (APP_DIR / "contact_inbox.txt").write_text(
        f"{payload.name} <{payload.email}>: {payload.message}\n", encoding="utf-8")
    return {"ok": True}


# BE/app.py
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "https://fintech-portfolio.pages.dev",
    "https://portfolio.yourdomain.com"  # when you add it
  ],
  allow_credentials=True,
  allow_methods=["GET","POST","OPTIONS"],
  allow_headers=["Content-Type"],
)


# --- Compound Interest API ---

from pydantic import Field


class CompoundInput(BaseModel):
    principal: float = Field(ge=0)
    rate_pct: float = Field(ge=0)          # annual rate in %
    years: float = Field(ge=0)
    compounds_per_year: int = Field(gt=0)
    contribution: float = Field(ge=0)      # recurring per-period

class CompoundPoint(BaseModel):
    t: float
    balance: float
    principal: float
    contributed: float
    interest: float

class CompoundOutput(BaseModel):
    final_value: float
    principal: float
    total_contributions: float
    total_interest: float
    points: List[CompoundPoint]

@app.post("/api/finance/compound", response_model=CompoundOutput)
def compound(payload: CompoundInput):
    P = payload.principal
    r = payload.rate_pct / 100.0
    n = payload.compounds_per_year
    t_years = payload.years
    contrib = payload.contribution

    periods = int(n * t_years)
    balance = P
    total_contrib = 0.0

    points: List[CompoundPoint] = []
    for k in range(periods + 1):
        t = k / n
        interest_val = balance - P - total_contrib
        points.append(CompoundPoint(
            t=t,
            balance=balance,
            principal=P,
            contributed=total_contrib,
            interest=interest_val
        ))
        # advance one period
        balance *= (1 + r / n)
        balance += contrib
        total_contrib += contrib

    final = points[-1].balance
    total_interest = final - P - total_contrib

    return CompoundOutput(
        final_value=final,
        principal=P,
        total_contributions=total_contrib,
        total_interest=total_interest,
        points=points
    )

@app.post("/api/finance/compound/plot")
def compound_plot(payload: CompoundInput):
    P = payload.principal
    r = payload.rate_pct / 100.0
    n = payload.compounds_per_year
    t_years = payload.years
    contrib = payload.contribution

    periods = int(n * t_years)
    balance = P
    total_contrib = 0.0

    times = []
    balances = []
    principals = []
    contribs = []
    interests = []

    for k in range(periods + 1):
        t = k / n
        interest_val = balance - P - total_contrib
        times.append(t)
        balances.append(balance)
        principals.append(P)
        contribs.append(total_contrib)
        interests.append(interest_val)

        balance *= (1 + r / n)
        balance += contrib
        total_contrib += contrib

    # ---- Make plot ----
    fig, ax = plt.subplots(figsize=(6, 4))

    base = [P + tc for tc in contribs]  # principal + total contributions at each time

    # Fill the base amount
    ax.fill_between(times, 0, base, alpha=0.35, color="#9ecae1", label="Principal + Contrib")

    # Fill the interest (gap between base and balance)
    ax.fill_between(times, base, balances, alpha=0.6, color="#fdd0a2", label="Interest")

    # Lines on top for clarity
    ax.plot(times, balances, color="blue", linewidth=2, label="Balance")
    ax.plot(times, base, color="green", linestyle="--", label="Principal + Contrib (line)")

    ax.set_title("Compound Interest Growth")
    ax.set_xlabel("Years")
    ax.set_ylabel("Balance (€)")

    # De-duplicate legend labels (optional)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False)

    ax.grid(True, alpha=0.3)


    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


# --- FX Converter API ---

from datetime import date, timedelta
import io, math
import httpx
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

FX_CURRENCIES = ["USD","EUR","GBP","JPY","AUD","CAD","CHF","CNY","SEK","NZD","MXN",
                 "SGD","HKD","NOK","KRW","TRY","INR","BRL","ZAR","AED","SAR","PLN",
                 "TWD","THB","DKK","MYR","IDR","PHP","CZK","HUF","ILS","CLP"]

class FXInput(BaseModel):
    amount: float
    base: str
    quote: str
    lookback_days: int

class FXOutput(BaseModel):
    last_date: str
    first_date: str
    points: int
    baseline_rate: float
    baseline_value: float
    final_value: float
    change_abs: float
    change_pct: float

@app.get("/api/fx/currencies", response_model=List[str])
def fx_currencies():
    return FX_CURRENCIES

async def _fetch_timeseries(base: str, quote: str, start_s: str, end_s: str):
    # Frankfurter API (no key): matches the PyScript version you shared
    url = f"https://api.frankfurter.dev/v1/{start_s}..{end_s}?base={base}&symbols={quote}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    # data: {"amount":1.0,"base":"USD","start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD","rates":{date:{quote:rate}}}
    rates = data.get("rates", {})
    items = sorted(rates.items(), key=lambda kv: kv[0])  # [(date, {quote:rate}), ...]
    xs, ys = [], []
    for d, one in items:
        v = one.get(quote)
        if isinstance(v, (int, float)) and math.isfinite(v):
            xs.append(d); ys.append(float(v))
    if len(xs) < 2:
        raise HTTPException(status_code=502, detail="Not enough FX data")
    return xs, ys

@app.post("/api/fx/convert", response_model=FXOutput)
async def fx_convert(payload: FXInput):
    if payload.amount <= 0 or payload.lookback_days < 2:
        raise HTTPException(status_code=400, detail="Invalid amount or lookback")
    if payload.base == payload.quote:
        raise HTTPException(status_code=400, detail="Base and quote must differ")
    end = date.today()
    start = end - timedelta(days=payload.lookback_days)
    start_s, end_s = start.isoformat(), end.isoformat()

    xs, rates = await _fetch_timeseries(payload.base, payload.quote, start_s, end_s)
    values = [payload.amount * r for r in rates]
    baseline_rate = rates[0]
    baseline_value = payload.amount * baseline_rate
    final_value = values[-1]
    change_abs = final_value - baseline_value
    change_pct = (change_abs / baseline_value) * 100 if baseline_value else 0.0

    return FXOutput(
        last_date=xs[-1], first_date=xs[0], points=len(xs),
        baseline_rate=baseline_rate, baseline_value=baseline_value,
        final_value=final_value, change_abs=change_abs, change_pct=change_pct
    )

@app.post("/api/fx/plot")
async def fx_plot(payload: FXInput):
    # Same validation
    if payload.amount <= 0 or payload.lookback_days < 2 or payload.base == payload.quote:
        raise HTTPException(status_code=400, detail="Bad inputs")
    end = date.today()
    start = end - timedelta(days=payload.lookback_days)
    start_s, end_s = start.isoformat(), end.isoformat()

    xs, rates = await _fetch_timeseries(payload.base, payload.quote, start_s, end_s)
    values = [payload.amount * r for r in rates]
    baseline_rate = rates[0]
    baseline_value = payload.amount * baseline_rate

    deltas = [v - baseline_value for v in values]
    X = list(range(len(xs)))
    y_baseline = [baseline_value] * len(values)
    y_actual = values

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Stacked fill: baseline (light) + gains/losses (green/red)
    ax.fill_between(X, 0, y_baseline, alpha=0.35, color="#9ecae1", label="Baseline value")

    pos = [max(0.0, d) for d in deltas]
    if any(p > 0 for p in pos):
        ax.fill_between(X, y_baseline, [b+p for b,p in zip(y_baseline, pos)],
                        where=[p>0 for p in pos], alpha=0.6, color="#34d399", label="Gain vs baseline")
    neg = [min(0.0, d) for d in deltas]
    if any(n < 0 for n in neg):
        ax.fill_between(X, y_baseline, [b+n for b,n in zip(y_baseline, neg)],
                        where=[n<0 for n in neg], alpha=0.5, color="#ef4444", label="Loss vs baseline")

    ax.plot(X, y_actual, linewidth=1.5, color="#1f77b4", label="Actual value")

    ax.set_title(f"{payload.amount:,.0f} {payload.base} in {payload.quote} over time")
    ax.set_ylabel(f"Value in {payload.quote}")
    ax.set_xlabel("Date")

    step = max(1, len(xs)//6)
    xticks = list(range(0, len(xs), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([xs[i] for i in xticks], rotation=30, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
# --- Stocks (Alpha Vantage: Monthly Adjusted) ---
import os, io, json, math
from datetime import date, timedelta, datetime
import httpx
import matplotlib.pyplot as plt

from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY")  # set this in Cloud Run/env
class StockInput(BaseModel):
    symbol: str
    period: str = Field(pattern="^(1mo|3mo|6mo|1y|5y|10y|ytd|max)$")

class StockSeriesPoint(BaseModel):
    date: str
    close: float

class StockSeriesOut(BaseModel):
    symbol: str
    first_date: str
    last_date: str
    rows: int
    period: str
    close_first: float
    close_last: float
    change_abs: float
    change_pct: float
    # We return just meta; plotting handled by /plot. (Add points if you want.)

_PERIOD_TO_DAYS = {
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
    "5y": 365*5, "10y": 365*10, "ytd": "ytd", "max": None,
}

def _filter_period(dates, closes, period):
    if _PERIOD_TO_DAYS[period] is None:
        return dates, closes
    if _PERIOD_TO_DAYS[period] == "ytd":
        start = date(date.today().year, 1, 1)
    else:
        start = date.today() - timedelta(days=_PERIOD_TO_DAYS[period])
    out_d, out_c = [], []
    for d, c in zip(dates, closes):
        if d >= start:
            out_d.append(d); out_c.append(c)
    return out_d, out_c

def _parse_alpha_series(js):
    KEY = "Monthly Adjusted Time Series"
    if KEY not in js:
        msg = (
            js.get("Note")
            or js.get("Information")
            or js.get("Error Message")
            or f"Unexpected response keys: {list(js.keys())}"
        )
        raise HTTPException(status_code=502, detail=msg)
    series = js[KEY]
    dates, closes = [], []
    for k, v in series.items():
        try:
            dt = datetime.strptime(k, "%Y-%m-%d").date()
            cv = float(v.get("4. close") or v.get("5. adjusted close") or v.get("4. Close", 0.0))
            dates.append(dt); closes.append(cv)
        except Exception:
            pass
    z = sorted(zip(dates, closes), key=lambda x: x[0])
    dates = [d for d,_ in z]; closes = [c for _,c in z]
    if len(dates) < 2:
        raise HTTPException(status_code=502, detail="Alpha Vantage returned too few rows.")
    return dates, closes

import time
from typing import Dict, Tuple, Any

_ALPHA_CACHE: Dict[str, Tuple[float, Any]] = {}
_ALPHA_TTL_SECONDS = 60  # 1 minute

async def _fetch_monthly_adjusted(symbol: str):
    if not ALPHA_KEY:
        raise HTTPException(status_code=500, detail="ALPHAVANTAGE_API_KEY not configured")

    now = time.time()
    hit = _ALPHA_CACHE.get(symbol)
    if hit and (now - hit[0]) < _ALPHA_TTL_SECONDS:
        return hit[1]

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={symbol}&apikey={ALPHA_KEY}"
    )
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()

    _ALPHA_CACHE[symbol] = (now, data)
    return data

@app.post("/api/stocks/series", response_model=StockSeriesOut)
async def stocks_series(payload: StockInput):
    #raise HTTPException(status_code=501, detail="app.py 455")
    sym = payload.symbol.strip().upper()
    js = await _fetch_monthly_adjusted(sym)
    dates, closes = _parse_alpha_series(js)
    dates, closes = _filter_period(dates, closes, payload.period)
    if len(dates) < 2:
        raise HTTPException(status_code=502, detail="No rows after filtering")
    first, last = dates[0], dates[-1]
    c0, c1 = closes[0], closes[-1]
    change_abs = c1 - c0
    change_pct = (change_abs / c0) * 100 if c0 else 0.0
    return StockSeriesOut(
        symbol=sym, first_date=str(first), last_date=str(last), rows=len(dates),
        period=payload.period, close_first=c0, close_last=c1,
        change_abs=change_abs, change_pct=change_pct
    )



@app.post("/api/stocks/plot")
async def stocks_plot(payload: StockInput):
    sym = payload.symbol.strip().upper()
    js = await _fetch_monthly_adjusted(sym)
    dates, closes = _parse_alpha_series(js)
    dates, closes = _filter_period(dates, closes, payload.period)

    X = list(range(len(dates)))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))

    # Filled area under the curve + line on top
    ax.fill_between(X, 0, closes, alpha=0.25, color="#9ecae1", label="Close (area)")
    ax.plot(X, closes, linewidth=1.6, color="#1f77b4", label="Close")

    ax.set_title(f"{sym} — {payload.period} (Monthly Adjusted)")
    ax.set_ylabel("Close")
    ax.set_xlabel("Date")

    step = max(1, len(dates)//6)
    xticks = list(range(0, len(dates), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(dates[i]) for i in xticks], rotation=30, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), frameon=False)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

# app.py (top-level, near other imports)
from fastapi.staticfiles import StaticFiles
from pathlib import Path



from pathlib import Path
from fastapi.staticfiles import StaticFiles

# mêmes dossiers que ci-dessus
APP_DIR = Path(__file__).parent
GRAPHS_DIR = APP_DIR / "backtrade" / "output" / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)




from pydantic import BaseModel, ConfigDict
from typing import Optional

class BacktestRequest(BaseModel):
    # Explicitly allow extra keys to be ignored
    model_config = ConfigDict(extra='ignore')

    symbol: str
    strategy: str
    start_period: str
    end_period: str
    starting_capital: float
    commission: float
    slippage: float




def _resolve_strategy(name: str):
    import importlib
    strats_mod = importlib.import_module("strats")
    lut = {cls.__name__: cls for cls in strats_mod.retall()}
    if name not in lut:
        raise HTTPException(status_code=400, detail=f"Unknown strategy name: {name}")
    return lut[name]

@app.get("/api/strategies")
def api_strategies():
    mod = importlib.import_module("strats")
    return {"items": [cls.__name__ for cls in mod.retall()]}

from fastapi import HTTPException

import os
import importlib
import traceback
from fastapi import HTTPException

@app.post("/api/backtest")
def api_backtest(req: BacktestRequest):
    print(f"[/api/backtest] symbol={req.symbol} strategy={req.strategy}")

    try:
        backtester = importlib.import_module("backtradercsvexport")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import backtester failed: {type(e).__name__}: {e}")

    try:
        StratCls = _resolve_strategy(req.strategy)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"_resolve_strategy failed: {type(e).__name__}: {e}")

    try:
        _clear_plots()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"_clear_plots failed: {type(e).__name__}: {e}")

    try:
        print("trying to run backtester.run_one()")
        result = backtester.run_one(
            req.symbol,
            StratCls,
            start_date=req.start_period,
            end_date=req.end_period,
            starting_capital=req.starting_capital,
            commission=req.commission,
            slippage=req.slippage,
        )
        if not result:
            raise HTTPException(
                status_code=502,
                detail="No result returned. Likely no data feed (make_feed=None) or zero bars in date range."
            )

    except HTTPException:
        raise

    except Exception as e:
        # Minimal extra context: exception type + where it blew up
        tb = traceback.extract_tb(e.__traceback__)
        last = tb[-1] if tb else None
        where = f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}" if last else "unknown location"

        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed ({type(e).__name__}) at {where}: {e}"
        )

    png_names = [
        "Calmar.png","MaxDrawdown_%.png","ProfitFactor.png","rnorm100_%.png",
        "SharpeAnnual.png","SharpeDaily.png","Sortino.png","SQN.png",
        "TimeDD_bars.png","TotalReturn_%.png","VWR.png","WinRate_%.png"
    ]
    charts = [f"/graphs/{n}" for n in png_names if (GRAPHS_DIR / n).exists()]

    plot_name = f"{req.symbol}_{req.strategy}.png"
    plot_path = GRAPHS_DIR / plot_name
    plot_url  = f"/graphs/{plot_name}" if plot_path.exists() else None

    if plot_url:
        charts.append(plot_url)

    return {"ok": True, "metrics": result, "charts": charts, "plot": plot_url}

    




from fastapi.responses import JSONResponse
from starlette.requests import Request

@app.exception_handler(Exception)
async def unhandled_exceptions(_: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc  # let FastAPI handle it normally

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

@app.get("/api/ping")
def ping():
    return {"ok": True}

def _clear_plots():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for p in GRAPHS_DIR.glob("*.png"):
        try:
            p.unlink()
        except Exception as e:
            raise HTTPException(500, p, "error : _clear_plots : image not found")



# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from hashlib import md5
import mimetypes



def _nocache_headers(path: Path):
    # Optional: strong ETag based on file content
    h = md5(path.read_bytes()).hexdigest()
    return {
        "Cache-Control": "no-store, max-age=0, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "ETag": h,
    }

@app.get("/graphs/{name}")
def get_graph(name: str):
    f = GRAPHS_DIR / name
    if not f.exists() or not f.is_file():
        raise HTTPException(404, "image not found")
    mt, _ = mimetypes.guess_type(str(f))
    return FileResponse(
        path=f,
        media_type=mt or "image/png",
        headers=_nocache_headers(f),
    )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT_DIR = APP_DIR.parent          # .../fintech-portfolio
FE_DIR = ROOT_DIR / "FE"           # .../fintech-portfolio/FE

if FE_DIR.exists():
    # Serve all FE files (html, css, js, images) under "/"
    app.mount(
        "/",
        StaticFiles(directory=FE_DIR, html=True),
        name="frontend",
    )

