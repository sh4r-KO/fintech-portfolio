from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import json
from pathlib import Path


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "projects.json"


app = FastAPI(title="Fintech Portfolio API", version="1.0.0")


# Allow local file hosting and common static hosts; adjust in production
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"]
,
allow_headers=["*"]
)

class Link(BaseModel):
    github: Optional[str] = None
    demo: Optional[str] = None
    paper: Optional[str] = None


class Metrics(BaseModel):
    rows_processed: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_savings_pct: Optional[float] = None


class Project(BaseModel):
    id: str
    slug: str
    title: str
    summary: str
    tags: List[str] = []
    tech: List[str] = []
    problem: Optional[str] = None
    approach: Optional[str] = None
    results: Optional[str] = None
    cover_image: Optional[str] = None
    links: Optional[Link] = None
    metrics: Optional[Metrics] = None


class Contact(BaseModel):
    name: str
    email: EmailStr
    message: str


# Load projects into memory (simple for demo; swap for DB later)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    PROJECTS = [Project(**p) for p in json.load(f)]
INDEX_BY_SLUG = {p.slug: p for p in PROJECTS}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/projects", response_model=List[Project])
def list_projects(q: Optional[str] = None, tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    items = PROJECTS
    if q:
        ql = q.lower()
        items = [p for p in items if ql in p.title.lower() or ql in p.summary.lower() or any(ql in t.lower() for t in p.tech + p.tags)]
    if tag:
        tl = tag.lower()
        items = [p for p in items if any(tl == t.lower() for t in p.tags)]
    return items[offset: offset + limit]


@app.get("/api/projects/{slug}", response_model=Project)
def get_project(slug: str):
    p = INDEX_BY_SLUG.get(slug)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return p


@app.post("/api/contact")
def contact(payload: Contact):
    # For now, just write to a local file. Replace with email/queue in prod.
    inbox = APP_DIR / "contact_inbox.txt"
    line = f"{payload.name} <{payload.email}>: {payload.message}\n"
    with open(inbox, "a", encoding="utf-8") as f:
        f.write(line)
    return {"ok": True}