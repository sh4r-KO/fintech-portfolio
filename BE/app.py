from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr,Field
from typing import List, Optional
from pathlib import Path
import json

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
import io
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

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
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(times, balances, label="Balance", color="blue")
    ax.plot(times, [P+tc for tc in contribs], label="Principal+Contrib", color="green", linestyle="--")
    ax.fill_between(times, [P+tc for tc in contribs], balances, alpha=0.3, color="orange", label="Interest")

    ax.set_title("Compound Interest Growth")
    ax.set_xlabel("Years")
    ax.set_ylabel("Balance (€)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
