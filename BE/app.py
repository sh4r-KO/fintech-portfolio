from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr,Field
from typing import List, Optional
from pathlib import Path
import json

import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "projects.json"

app = FastAPI(title="Fintech Portfolio API", version="1.0.0")
#app.add_middleware(
#    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
#    allow_methods=["*"], allow_headers=["*"]
#)

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
