"""Portfolio Pulse API: AI news intelligence for portfolio managers."""
import json
import os
import re
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models import AnalyzeRequest
from .pipeline import run_analysis

app = FastAPI(
    title="Portfolio Pulse",
    description="AI news intelligence for portfolio managers: bear-vs-bull agent debate over filtered, material company news.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict] = {}


def _set(job_id: str, **kw) -> None:
    if job_id in jobs:
        jobs[job_id].update(kw)


def _run_job(job_id: str, ticker: str, company_name: str, query: str, days: int) -> None:
    def progress(stage: str, pct: int, message: str) -> None:
        _set(job_id, status=stage, progress=pct, stage=stage, message=message)

    try:
        run_analysis(job_id, ticker, company_name, query, days, progress)
        _set(job_id, status="done", progress=100, stage="done", message="Analysis complete.")
    except Exception as e:  # never leave a job hanging
        _set(job_id, status="error", stage="error",
             message="Analysis failed.", error=str(e)[:600])


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, bg: BackgroundTasks):
    ticker = req.ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "job_id": job_id, "status": "queued", "progress": 0,
        "stage": "queued", "message": "Queued.", "error": "",
    }
    bg.add_task(_run_job, job_id, ticker, req.company_name.strip(), req.query.strip(), req.days)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
def get_result(job_id: str):
    path = Path(settings.data_dir) / "results" / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not ready or unknown job id.")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "portfolio-pulse",
        "live_news": settings.live_news,
        "live_llm": settings.live_llm,
        "llm_provider": settings.llm_provider,
        "demo_mode": not (settings.live_news and settings.live_llm),
    }


@app.get("/sitemap.xml", response_class=PlainTextResponse)
def sitemap(request: Request):
    base = str(request.base_url).rstrip("/")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{base}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n"
        "</urlset>"
    )


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots(request: Request):
    base = str(request.base_url).rstrip("/")
    return f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n"


STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
