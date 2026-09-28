"""News fetching clients.

Strategy (validated Sept 2026):
- Finnhub is the PRIMARY fetcher: 60 req/min, no daily cap, months of history,
  so it satisfies the 25-day lookback. It returns no sentiment -> we score that
  ourselves in preprocessing (finance-tuned lexicon).
- Alpha Vantage NEWS_SENTIMENT is the ENRICHMENT source: finance-tuned
  per-ticker sentiment + relevance scores + topic tags. Free tier only returns
  current news (ignores time_from), so it cannot be the primary.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, date

import requests

TIMEOUT = 30


@dataclass
class RawArticle:
    title: str
    summary: str
    url: str
    source: str
    source_domain: str
    published_at: datetime
    tickers: list[str] = field(default_factory=list)
    ticker_relevance: dict = field(default_factory=dict)  # ticker -> 0..1
    ticker_sentiment: dict = field(default_factory=dict)  # ticker -> -1..1
    overall_sentiment: float | None = None
    topics: list[tuple[str, float]] = field(default_factory=list)
    origin: str = ""


class NewsAPIError(RuntimeError):
    pass


class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def company_news(self, symbol: str, from_date: date, to_date: date) -> list[RawArticle]:
        params = {
            "symbol": symbol.upper(),
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.api_key,
        }
        r = requests.get(f"{self.BASE}/company-news", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):  # error payload
            raise NewsAPIError(f"Finnhub: {data.get('error', data)}")
        articles: list[RawArticle] = []
        for item in data:
            try:
                ts = int(item.get("datetime", 0))
                published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(timezone.utc)
            except (TypeError, ValueError):
                published = datetime.now(timezone.utc)
            related = [t.strip().upper() for t in str(item.get("related", "") or "").split(",") if t.strip()]
            articles.append(
                RawArticle(
                    title=str(item.get("headline", "") or "").strip(),
                    summary=str(item.get("summary", "") or "").strip(),
                    url=str(item.get("url", "") or "").strip(),
                    source=str(item.get("source", "") or "").strip(),
                    source_domain="",
                    published_at=published,
                    tickers=related,
                    ticker_relevance={t: 0.9 for t in related},
                    origin="finnhub",
                )
            )
        return [a for a in articles if a.title]


class AlphaVantageClient:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def news_sentiment(self, tickers: list[str], limit: int = 200) -> list[RawArticle]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(t.upper() for t in tickers),
            "apikey": self.api_key,
            "limit": limit,
            "sort": "LATEST",
        }
        r = requests.get(self.BASE, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if "feed" not in data:
            note = data.get("Note") or data.get("Information") or str(data)[:200]
            raise NewsAPIError(f"Alpha Vantage: {note}")
        articles: list[RawArticle] = []
        for item in data["feed"]:
            try:
                published = datetime.strptime(
                    item.get("time_published", ""), "%Y%m%dT%H%M%S"
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                published = datetime.now(timezone.utc)
            tickers_list, rel, sent = [], {}, {}
            for ts in item.get("ticker_sentiment", []) or []:
                t = str(ts.get("ticker", "")).upper()
                if not t:
                    continue
                tickers_list.append(t)
                try:
                    rel[t] = float(ts.get("relevance_score", 0))
                except (TypeError, ValueError):
                    rel[t] = 0.0
                try:
                    sent[t] = float(ts.get("ticker_sentiment_score", 0))
                except (TypeError, ValueError):
                    sent[t] = 0.0
            topics = []
            for tp in item.get("topics", []) or []:
                try:
                    topics.append((str(tp.get("topic", "")), float(tp.get("relevance_score", 0))))
                except (TypeError, ValueError):
                    continue
            try:
                overall = float(item.get("overall_sentiment_score", 0))
            except (TypeError, ValueError):
                overall = None
            domain = str(item.get("source_domain", "") or "").lower().replace("www.", "")
            articles.append(
                RawArticle(
                    title=str(item.get("title", "") or "").strip(),
                    summary=str(item.get("summary", "") or "").strip(),
                    url=str(item.get("url", "") or "").strip(),
                    source=str(item.get("source", "") or "").strip(),
                    source_domain=domain,
                    published_at=published,
                    tickers=tickers_list,
                    ticker_relevance=rel,
                    ticker_sentiment=sent,
                    overall_sentiment=overall,
                    topics=topics,
                    origin="alphavantage",
                )
            )
        return [a for a in articles if a.title]


def _norm_url(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://(www\.)?", "", u)
    # strip fragment, then drop only tracking params (keep ?id= etc. — some
    # providers put the article id in the query string)
    u = u.split("#")[0]
    if "?" in u:
        base, qs = u.split("?", 1)
        kept = [p for p in qs.split("&")
                if p and not p.startswith(("utm_", "fbclid", "gclid", "mc_cid", "mc_eid"))]
        u = base + ("?" + "&".join(sorted(kept)) if kept else "")
    return u.rstrip("/")


def dedupe(articles: list[RawArticle]) -> list[RawArticle]:
    """Dedupe by normalized URL, falling back to title hash. Keeps first seen."""
    seen: set[str] = set()
    out: list[RawArticle] = []
    for a in articles:
        key = _norm_url(a.url) if a.url else "t:" + hashlib.sha1(a.title.lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def fetch_all(settings, ticker: str, from_date: date, to_date: date) -> tuple[list[RawArticle], list[str]]:
    """Fetch from all configured sources. Never raises: returns what succeeded."""
    articles: list[RawArticle] = []
    sources: list[str] = []
    if settings.finnhub_api_key:
        try:
            fh = FinnhubClient(settings.finnhub_api_key).company_news(ticker, from_date, to_date)
            articles.extend(fh)
            sources.append(f"Finnhub ({len(fh)})")
        except Exception as e:
            sources.append(f"Finnhub error: {str(e)[:80]}")
    if settings.alpha_vantage_api_key:
        try:
            av = AlphaVantageClient(settings.alpha_vantage_api_key).news_sentiment([ticker])
            articles.extend(av)
            sources.append(f"Alpha Vantage ({len(av)})")
        except Exception as e:
            sources.append(f"Alpha Vantage error: {str(e)[:80]}")
    return dedupe(articles), sources
