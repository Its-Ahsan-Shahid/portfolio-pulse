# Portfolio Pulse

**AI news intelligence for portfolio managers.** Enter a company + ticker + your question
(e.g. *"What is the financial risk for this company?"*), and Portfolio Pulse:

1. **Fetches** the last N days of company news (Finnhub primary, Alpha Vantage enrichment)
2. **Preprocesses** — entity relevance (NER + ticker matching), financial-relevance
   classification (sentiment ≠ importance), and materiality scoring with a
   recency boost for the last 5 days
3. **Embeds** material news into a per-analysis ChromaDB vector store
4. Runs a **bear-vs-bull agent debate** (bull hunts upside, bear hunts downside),
   adjudicated by a **moderator agent** that answers your actual question
5. Renders a **dashboard**: verdict, moderator brief, 4 charts, agent cases, cited news feed

LLM default is **Groq** (free tier); swap to **Grok (xAI)** with one env var.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # then fill in your keys
uvicorn backend.app.main:app --reload --port 8000
```

Open http://localhost:8000

### Environment variables

| Var | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | for live agents | Bear/bull/moderator LLM (free at console.groq.com) |
| `FINNHUB_API_KEY` | for live news | Primary news fetch, 25-day lookback (free at finnhub.io) |
| `ALPHA_VANTAGE_API_KEY` | optional | Finance-tuned sentiment enrichment (free, 25 calls/day) |
| `XAI_API_KEY` + `LLM_PROVIDER=xai` | optional | Use Grok instead of Groq for agents |
| `PRIORITY_DAYS` | no (default 5) | Recency priority window |
| `MATERIALITY_THRESHOLD` | no (default 0.45) | Filter cutoff |
| `DATA_DIR` | no | Where results + vector DB live |

With no keys at all, the app runs in clearly-labeled **demo mode**
(synthetic news + heuristic agents) so the UI is testable offline.

## API

- `POST /api/analyze` `{ticker, company_name, query, days}` → full analysis JSON
  (verdict, bull/bear cases, charts, cited news). Runs synchronously in one
  request — no polling, no job store.
- `GET /health`, `/api/health`, `/sitemap.xml`, `/robots.txt`

## Deploy — Vercel (free, no credit card)

The app is serverless-ready: one FastAPI application served as a Vercel
Python function (`api/index.py`), with `vercel.json` routing every path to it
and `maxDuration: 300` so full analyses comfortably finish.

1. Push this repo to GitHub, then **Import** it at https://vercel.com/new
   (sign in with GitHub — no credit card required).
2. In the Vercel project → **Settings → Environment Variables**, add:
   `GROQ_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`.
3. **Deploy.** The dashboard is live at `https://<project>.vercel.app`.

> Render's free tier now requires a credit card for verification, and
> Hugging Face paywalled Docker/Gradio Spaces — the `Dockerfile.render`,
> `render.yaml`, and `backend/requirements-render.txt` remain in the repo as
> reference, but Vercel is the recommended free target.

## Deploy — Docker (any host)

`Dockerfile.render` builds the lean image (no torch/spacy/chromadb):
`docker build -f Dockerfile.render -t portfolio-pulse .`
Run with `-e GROQ_API_KEY=… -e FINNHUB_API_KEY=… -e ALPHA_VANTAGE_API_KEY=… -p 8000:8000`.
Health check: `/api/health`.

## Project layout

```
backend/app/
  main.py             FastAPI app (synchronous /api/analyze), SEO routes, static UI
  pipeline.py         orchestration + demo fixture + heuristic fallback
  news_clients.py     Finnhub + Alpha Vantage fetchers
  preprocess.py       entity relevance, financial taxonomy, materiality
  embeddings_store.py ChromaDB (+ TF-IDF fallback) per-job vector store
  agents.py           bull / bear / moderator prompts + OpenAI-compatible client
  static/             dashboard (index.html, styles.css, app.js)
```

## Disclaimer

For informational purposes only — not financial advice.
