"""Bear vs Bull vs Moderator agents.

Convention (market standard): the BULL hunts upside/positive evidence, the
BEAR hunts downside/negative evidence. Both argue ONLY from retrieved,
pre-filtered material news and must cite article titles. The MODERATOR weighs
both cases and answers the portfolio manager's actual query.

The LLM layer talks to any OpenAI-compatible endpoint. Default: Groq (free).
Swap LLM_PROVIDER=xai + XAI_API_KEY for Grok with zero code changes.
"""
from __future__ import annotations

import json
import re

from .embeddings_store import BaseStore

DISCLAIMER = "For informational purposes only — not financial advice."


# ------------------------------------------------------------ LLM client ---

class LLMClient:
    def __init__(self, base_url: str, api_key: str, moderator_model: str, agent_model: str):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.moderator_model = moderator_model
        self.agent_model = agent_model

    def chat_json(self, system: str, user: str, model: str, max_tokens: int = 1600) -> dict:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return _safe_json(resp.choices[0].message.content or "")


def _safe_json(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"_raw": text}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {"_raw": text}
        except Exception:
            pass
    return {"_raw": text}


# --------------------------------------------------------------- agents ---

BULL_SYSTEM = """You are the BULL analyst on an institutional investment committee.
Your job: build the strongest evidence-backed BULLISH case for {company} ({ticker})
regarding the portfolio manager's question: "{query}".

Rules:
- Use ONLY the evidence provided below. Do not invent facts, numbers, or events.
- Every substantive claim must cite the article title it comes from.
- Steelman the upside: catalysts, beats, guidance raises, margin expansion,
  capital returns, de-risking events. Acknowledge the strongest 1-2 bearish
  counterpoints honestly at the end.
- Return STRICT JSON with exactly these keys:
  {{"thesis": str (2-3 sentences), "score": float 0..1 (bullish conviction),
    "confidence": float 0..1, "bullets": [4-6 concise strings],
    "evidence": [{{"title": str, "why": str}} up to 6],
    "acknowledged_risks": [1-2 strings]}}"""

BEAR_SYSTEM = """You are the BEAR analyst on an institutional investment committee.
Your job: build the strongest evidence-backed BEARISH case for {company} ({ticker})
regarding the portfolio manager's question: "{query}".

Rules:
- Use ONLY the evidence provided below. Do not invent facts, numbers, or events.
- Every substantive claim must cite the article title it comes from.
- Steelman the downside: misses, guidance cuts, margin compression, rising costs,
  regulatory/legal overhang, balance-sheet stress, insider/analyst negativity.
  Acknowledge the strongest 1-2 bullish counterpoints honestly at the end.
- Return STRICT JSON with exactly these keys:
  {{"thesis": str (2-3 sentences), "score": float 0..1 (bearish conviction),
    "confidence": float 0..1, "bullets": [4-6 concise strings],
    "evidence": [{{"title": str, "why": str}} up to 6],
    "acknowledged_risks": [1-2 strings]}}"""

MODERATOR_SYSTEM = """You are the MODERATOR of an institutional investment committee.
You have received a BULL case and a BEAR case about {company} ({ticker}), built
from the same filtered news evidence. The portfolio manager's question is: "{query}".

Your job:
- Weigh both cases on evidence quality, not rhetoric. Discount claims with weak
  or single-source evidence. Note where both sides agree (that is high-signal).
- Consider base rates: single news cycles rarely change intrinsic value; repeated
  fundamental signals do.
- Give a direct answer to the manager's question.
- Return STRICT JSON with exactly these keys:
  {{"verdict": "bullish" | "bearish" | "neutral", "confidence": float 0..1,
    "summary": str (2-3 sentences), "bullets": [5-7 executive takeaways],
    "risks": [3-5 strings], "watch_items": [2-4 strings],
    "query_answer": str (direct answer to the manager's question)}}"""


def _evidence_block(hits) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        m = h.metadata or {}
        src = m.get("source", "unknown")
        date = str(m.get("published_at", ""))[:10]
        mat = m.get("materiality", "")
        lines.append(f"[{i}] ({src}, {date}, materiality {mat}) {m.get('title', '')}\n{h.text[:600]}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


def _normalize_case(raw: dict, side: str) -> dict:
    def _f(v, d=0.5):
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return d

    ev = raw.get("evidence") or []
    evidence = [
        {"title": str(e.get("title", ""))[:160], "why": str(e.get("why", ""))[:280]}
        for e in ev if isinstance(e, dict)
    ][:6]
    return {
        "score": _f(raw.get("score")),
        "confidence": _f(raw.get("confidence")),
        "thesis": str(raw.get("thesis", ""))[:800],
        "bullets": [str(b)[:280] for b in (raw.get("bullets") or [])][:6],
        "evidence": evidence,
        "acknowledged_risks": [str(r)[:280] for r in (raw.get("acknowledged_risks") or [])][:3],
        "side": side,
    }


def run_bull(llm: LLMClient, store: BaseStore, company: str, ticker: str, query: str) -> dict:
    hits = store.query(
        f"{company} {ticker} {query} positive upside growth beat raised guidance catalyst", n=10
    )
    raw = llm.chat_json(
        BULL_SYSTEM.format(company=company, ticker=ticker, query=query or "overall outlook"),
        "EVIDENCE (pre-filtered material news):\n\n" + _evidence_block(hits),
        model=llm.agent_model,
    )
    return _normalize_case(raw, "bull")


def run_bear(llm: LLMClient, store: BaseStore, company: str, ticker: str, query: str) -> dict:
    hits = store.query(
        f"{company} {ticker} {query} risk downside miss cut downgrade lawsuit loss threat", n=10
    )
    raw = llm.chat_json(
        BEAR_SYSTEM.format(company=company, ticker=ticker, query=query or "overall outlook"),
        "EVIDENCE (pre-filtered material news):\n\n" + _evidence_block(hits),
        model=llm.agent_model,
    )
    return _normalize_case(raw, "bear")


def _normalize_verdict(raw: dict) -> dict:
    def _f(v, d=0.5):
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except (TypeError, ValueError):
            return d

    verdict = str(raw.get("verdict", "neutral")).lower()
    if verdict not in ("bullish", "bearish", "neutral"):
        verdict = "neutral"
    return {
        "label": verdict,
        "confidence": _f(raw.get("confidence")),
        "summary": str(raw.get("summary", ""))[:600],
        "bullets": [str(b)[:300] for b in (raw.get("bullets") or [])][:7],
        "risks": [str(r)[:300] for r in (raw.get("risks") or [])][:5],
        "watch_items": [str(w)[:300] for w in (raw.get("watch_items") or [])][:4],
        "query_answer": str(raw.get("query_answer", ""))[:800],
        "disclaimer": DISCLAIMER,
    }


def run_moderator(
    llm: LLMClient,
    company: str,
    ticker: str,
    query: str,
    bull: dict,
    bear: dict,
    n_material: int,
    n_relevant: int,
) -> dict:
    user = (
        f"Portfolio manager's question: \"{query or 'overall outlook'}\"\n"
        f"Evidence base: {n_material} material articles (from {n_relevant} relevant).\n\n"
        f"BULL CASE (conviction {bull['score']:.2f}, confidence {bull['confidence']:.2f}):\n"
        f"Thesis: {bull['thesis']}\nBullets:\n- " + "\n- ".join(bull["bullets"]) + "\n\n"
        f"BEAR CASE (conviction {bear['score']:.2f}, confidence {bear['confidence']:.2f}):\n"
        f"Thesis: {bear['thesis']}\nBullets:\n- " + "\n- ".join(bear["bullets"])
    )
    raw = llm.chat_json(
        MODERATOR_SYSTEM.format(company=company, ticker=ticker, query=query or "overall outlook"),
        user,
        model=llm.moderator_model,
        max_tokens=1800,
    )
    return _normalize_verdict(raw)
