"""Orchestration: fetch -> preprocess -> embed -> agents -> assemble result.

Runs as a background task. Progress is reported through a callback.
If API keys are missing, clearly-labeled demo/heuristic paths keep the app
usable: demo news when no news keys, heuristic agents when no LLM key.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from . import agents as agent_mod
from .agents import DISCLAIMER
from .config import settings
from .embeddings_store import build_store, doc_text_for
from .news_clients import RawArticle, fetch_all
from .preprocess import preprocess


# ------------------------------------------------------- demo fixture ---

def demo_articles(company: str, ticker: str, days: int) -> list[RawArticle]:
    """Clearly-labeled synthetic articles so the UI works with zero API keys."""
    now = datetime.now(timezone.utc)
    t = ticker.upper()
    templates = [
        (f"{company} beats quarterly earnings estimates as services revenue jumps 14%", 0.7, "earnings", 2),
        (f"{company} raises full-year guidance citing strong enterprise demand", 0.6, "guidance", 4),
        (f"{company} announces $5B share buyback program and dividend hike", 0.55, "capital", 6),
        (f"Analysts upgrade {company} to Overweight, lift price target 18%", 0.5, "analyst", 3),
        (f"{company} cuts operating expenses 20% in efficiency drive", 0.1, "operations", 5),
        (f"{company} unveils next-generation product line for holiday quarter", 0.35, "product", 8),
        (f"Regulator opens antitrust review into {company}'s app store practices", -0.55, "regulatory", 7),
        (f"{company} misses cloud revenue expectations; shares slide after hours", -0.6, "earnings", 9),
        (f"{company} announces 4,000 layoffs as part of restructuring plan", -0.4, "operations", 11),
        (f"Supply chain disruptions hit {company}'s flagship device production", -0.35, "operations", 13),
        (f"{company} names new CFO from rival amid leadership reshuffle", 0.05, "leadership", 15),
        (f"Macro headwinds: analysts debate {company}'s exposure to rate outlook", -0.1, "macro", 18),
    ]
    articles = []
    for i, (title, sent, _cat, age) in enumerate(templates):
        if age > days:
            continue
        articles.append(
            RawArticle(
                title=title,
                summary=title + " — demo wire summary for interface testing.",
                url=f"https://example.com/demo/{t.lower()}-{i}",
                source="Demo Wire",
                source_domain="example.com",
                published_at=now - timedelta(days=age, hours=i),
                tickers=[t],
                ticker_relevance={t: 0.95},
                ticker_sentiment={t: sent},
                overall_sentiment=sent,
                origin="demo",
            )
        )
    return articles


# ------------------------------------------------- heuristic fallback ---

def _heuristic_cases(material, company: str, query: str) -> tuple[dict, dict, dict]:
    """Rule-based bull/bear/moderator used when no LLM key is configured."""
    pos = sorted([m for m in material if m.sentiment_score > 0.15],
                 key=lambda m: m.sentiment_score * m.materiality, reverse=True)[:4]
    neg = sorted([m for m in material if m.sentiment_score < -0.15],
                 key=lambda m: -m.sentiment_score * m.materiality, reverse=True)[:4]

    def case(items, side):
        score = 0.5
        if items:
            score = max(0.05, min(0.95, 0.5 + 0.5 * sum(m.sentiment_score * m.materiality for m in items) / len(items)))
        word = "upside" if side == "bull" else "downside"
        return {
            "score": round(score, 3),
            "confidence": 0.5,
            "thesis": (
                f"Heuristic {side} case for {company}: {len(items)} material {word} "
                f"signals identified from filtered news (LLM agents disabled — no API key)."
            ),
            "bullets": [m.article.title[:280] for m in items[:5]] or ["No strong directional signals in filtered news."],
            "evidence": [{"title": m.article.title[:160], "why": f"materiality {m.materiality}"} for m in items[:5]],
            "acknowledged_risks": [],
            "side": side,
            "heuristic": True,
        }

    bull, bear = case(pos, "bull"), case(neg, "bear")
    spread = bull["score"] - bear["score"]
    label = "bullish" if spread > 0.12 else ("bearish" if spread < -0.12 else "neutral")
    verdict = {
        "label": label,
        "confidence": round(min(0.9, 0.5 + abs(spread)), 3),
        "summary": (
            f"Heuristic verdict for {company}: {label} (heuristic mode — connect an LLM "
            f"key for full agent debate). Bull score {bull['score']:.2f} vs bear {bear['score']:.2f}."
        ),
        "bullets": [f"Top signal: {m.article.title[:240]}" for m in (material[:5])],
        "risks": [m.article.title[:240] for m in neg[:3]],
        "watch_items": ["Connect Groq/xAI API key for full bull-vs-bear agent debate."],
        "query_answer": f"Heuristic read on '{query or 'outlook'}': {label}.",
        "disclaimer": DISCLAIMER,
        "heuristic": True,
    }
    return bull, bear, verdict


# ------------------------------------------------------------ charts ---

def build_charts(relevant, material, days: int, priority_days: int) -> dict:
    now = datetime.now(timezone.utc)
    today = now.date()
    from_day = today - timedelta(days=days - 1)

    cat_counter: Counter = Counter()
    for m in material:
        for c in m.categories:
            cat_counter[c] += 1
    category_breakdown = [
        {"label": k, "count": v} for k, v in cat_counter.most_common(8)
    ]

    by_day: dict[date, list] = {}
    for r in relevant:
        d = r.article.published_at.date()
        if from_day <= d <= today:
            by_day.setdefault(d, []).append(r)
    timeline = []
    d = from_day
    while d <= today:
        items = by_day.get(d, [])
        mat_n = sum(1 for r in items if r.materiality >= settings.materiality_threshold)
        avg = round(sum(r.sentiment_score for r in items) / len(items), 3) if items else 0.0
        timeline.append({
            "date": d.isoformat(),
            "count": len(items),
            "avg_sentiment": avg,
            "material_count": mat_n,
            "priority": (today - d).days < priority_days,
        })
        d += timedelta(days=1)

    buckets = ["0.45–0.60", "0.60–0.75", "0.75–0.90", "0.90–1.00"]
    dist = [0, 0, 0, 0]
    for m in material:
        v = m.materiality
        dist[0 if v < 0.60 else 1 if v < 0.75 else 2 if v < 0.90 else 3] += 1
    materiality_dist = [{"bucket": b, "count": c} for b, c in zip(buckets, dist)]

    return {
        "category_breakdown": category_breakdown,
        "timeline": timeline,
        "materiality_dist": materiality_dist,
    }


# --------------------------------------------------------------- main ---

def run_analysis(job_id: str, ticker: str, company_name: str, query: str,
                 days: int, progress) -> dict:
    s = settings
    ticker = ticker.upper()
    today = datetime.now(timezone.utc).date()
    from_day = today - timedelta(days=days - 1)
    demo_news = not s.live_news

    # ---- 1. fetch ---------------------------------------------------------
    progress("fetching", 8, "Fetching news from configured sources…")
    if demo_news:
        articles = demo_articles(company_name, ticker, days)
        sources = ["Demo Wire (no news API keys configured)"]
    else:
        articles, sources = fetch_all(s, ticker, from_day, today)
    progress("fetching", 28, f"Fetched {len(articles)} articles. Deduplicating…")

    # ---- 2. preprocess ----------------------------------------------------
    progress("preprocessing", 38, "Running entity relevance + NER filter…")
    relevant, material = preprocess(
        articles, ticker, company_name, days, s.priority_days, s.materiality_threshold
    )
    progress("preprocessing", 55,
             f"{len(relevant)} relevant, {len(material)} material articles after filtering.")

    # ---- 3. embed ---------------------------------------------------------
    progress("embedding", 62, "Embedding material news into vector store…")
    store, vec_backend = build_store(
        os.path.join(s.data_dir, "chroma"), f"pp-{job_id}"
    )
    docs = []
    for i, m in enumerate(material):
        a = m.article
        docs.append({
            "id": f"doc-{i}",
            "text": doc_text_for(m),
            "metadata": {
                "title": a.title, "source": a.source or a.source_domain,
                "published_at": a.published_at.isoformat(),
                "materiality": m.materiality, "sentiment_label": m.sentiment_label,
                "url": a.url,
            },
        })
    store.add(docs)
    progress("embedding", 70, f"Indexed {len(docs)} articles ({vec_backend}).")

    # ---- 4. agents --------------------------------------------------------
    progress("agents", 76, "Bull agent building the upside case…")
    heuristic = not s.live_llm
    if heuristic:
        bull, bear, verdict = _heuristic_cases(material, company_name, query)
        models = {"provider": "heuristic", "note": "no LLM API key configured"}
    else:
        base_url, api_key, mod_model, agent_model = s.llm_endpoint()
        llm = agent_mod.LLMClient(base_url, api_key, mod_model, agent_model)
        bull = agent_mod.run_bull(llm, store, company_name, ticker, query)
        progress("agents", 84, "Bear agent building the downside case…")
        bear = agent_mod.run_bear(llm, store, company_name, ticker, query)
        progress("agents", 92, "Moderator weighing both cases…")
        verdict = agent_mod.run_moderator(
            llm, company_name, ticker, query, bull, bear, len(material), len(relevant)
        )
        models = {"provider": s.llm_provider, "moderator": mod_model, "agent": agent_model}
    store.close()

    # ---- 5. assemble ------------------------------------------------------
    progress("finalizing", 97, "Assembling dashboard…")
    charts = build_charts(relevant, material, days, s.priority_days)
    charts["agent_scores"] = {"bull": bull["score"], "bear": bear["score"]}

    news_out = []
    for m in material[:40]:
        a = m.article
        news_out.append({
            "title": a.title, "summary": a.summary, "url": a.url,
            "source": a.source or a.source_domain,
            "published_at": a.published_at.isoformat(),
            "sentiment_label": m.sentiment_label, "sentiment_score": m.sentiment_score,
            "relevance": m.relevance, "financial_relevance": m.financial_relevance,
            "materiality": m.materiality, "categories": m.categories,
            "priority": m.priority,
        })

    result = {
        "job_id": job_id,
        "ticker": ticker,
        "company_name": company_name,
        "query": query,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "fetched": len(articles),
            "relevant": len(relevant),
            "material": len(material),
            "date_from": from_day.isoformat(),
            "date_to": today.isoformat(),
        },
        "verdict": verdict,
        "bull_case": bull,
        "bear_case": bear,
        "charts": charts,
        "news": news_out,
        "meta": {
            "sources_used": sources,
            "models": models,
            "demo_mode": bool(demo_news or heuristic),
            "vector_backend": vec_backend,
            "disclaimer": DISCLAIMER,
        },
    }

    out_dir = os.path.join(s.data_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{job_id}.json"), "w") as f:
        json.dump(result, f)

    progress("done", 100, "Analysis complete.")
    return result
