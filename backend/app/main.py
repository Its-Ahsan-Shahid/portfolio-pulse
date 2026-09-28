"""Portfolio Pulse API: AI news intelligence for portfolio managers.

Serverless-friendly design: POST /api/analyze runs the full pipeline
synchronously and returns the complete result in one response. No background
tasks, no in-memory job store, no disk persistence — nothing that assumes a
long-lived process.
"""
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
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


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    ticker = req.ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")
    try:
        result = run_analysis(ticker, req.company_name.strip(), req.query.strip(), req.days)
    except Exception as e:  # never leak a stack trace; keep the message short
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:300]}")
    return JSONResponse(result)


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
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
