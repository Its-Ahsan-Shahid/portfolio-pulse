"""Preprocessing: the PM-grade filtering layer.

Three problems solved here, in order:

1. ENTITY RELEVANCE — is this article actually about the requested company?
   Signals: Alpha Vantage per-ticker relevance_score (primary), Finnhub
   `related` tickers, company-name token/fuzzy match, spaCy ORG NER (optional).

2. FINANCIAL RELEVANCE — does it matter to a portfolio manager?
   Sentiment != importance. "Athlete wins gold" is positive but irrelevant;
   "opex cut 20%" is neutral but critical. A finance taxonomy scores each
   article 0..1 by PM-relevant category weight.

3. MATERIALITY — blended score (relevance, financial weight, recency with a
   priority boost for the last N days, source authority). Only articles above
   threshold reach the embedding store and the agents.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .news_clients import RawArticle

# ---------------------------------------------------------------- taxonomy ---

# weight: how much a PM should care about this category (0..1)
FINANCE_TAXONOMY: dict[str, dict] = {
    "earnings": {"weight": 1.0, "keywords": [
        "earnings", "revenue", "eps", "ebitda", "ebit", "net income", "net loss",
        "gross margin", "operating margin", "profit", "loss", "beat estimates",
        "missed estimates", "quarterly results", "full-year", "top line", "bottom line"]},
    "guidance": {"weight": 1.0, "keywords": [
        "guidance", "outlook", "forecast", "expects", "raises outlook", "cuts outlook",
        "lowers forecast", "raises forecast", "projection", "forward looking"]},
    "m&a": {"weight": 1.0, "keywords": [
        "merger", "acquisition", "acquire", "acquiring", "takeover", "buyout",
        "stake", "divest", "divestiture", "spin-off", "spinoff", "merges with"]},
    "capital": {"weight": 0.95, "keywords": [
        "dividend", "buyback", "share repurchase", "stock split", "share offering",
        "debt", "bond", "credit facility", "financing", "ipo", "secondary offering",
        "leverage", "refinance", "credit rating"]},
    "regulatory": {"weight": 1.0, "keywords": [
        "sec", "lawsuit", "sues", "sued", "fine", "fined", "settlement", "antitrust",
        "investigation", "regulator", "doj", "fda approval", "fda", "sanction",
        "compliance", "subpoena", "class action"]},
    "leadership": {"weight": 0.7, "keywords": [
        "ceo", "cfo", "coo", "resigns", "resignation", "appointed", "named ceo",
        "board of directors", "activist investor", "proxy", "chairman", "succession"]},
    "product": {"weight": 0.6, "keywords": [
        "launches", "launch", "unveils", "recall", "patent", "clinical trial",
        "product line", "flagship"]},
    "macro": {"weight": 0.7, "keywords": [
        "federal reserve", "interest rate", "rate cut", "rate hike", "inflation",
        "recession", "gdp", "tariff", "trade war", "unemployment", "cpi"]},
    "analyst": {"weight": 0.75, "keywords": [
        "upgrade", "upgraded", "downgrade", "downgraded", "price target",
        "analyst", "rating", "initiated coverage", "outperform", "underperform"]},
    "operations": {"weight": 0.85, "keywords": [
        "layoff", "layoffs", "job cuts", "hiring freeze", "strike", "factory",
        "supply chain", "opex", "operating expenses", "cost cut", "cost-cutting",
        "restructuring", "headcount", "outsourc", "shutdown", "capacity"]},
}

# Alpha Vantage topic names -> our categories
AV_TOPIC_MAP = {
    "earnings": "earnings",
    "mergers_and_acquisitions": "m&a",
    "ipo": "capital",
    "financial_markets": "macro",
    "economy_fiscal": "macro",
    "economy_monetary": "macro",
    "economy_macro": "macro",
    "technology": "product",
    "finance": "capital",
    "life_sciences": "product",
    "manufacturing": "operations",
    "real_estate": "capital",
    "retail_wholesale": "operations",
    "energy_transportation": "operations",
}

_NUMERIC_RE = re.compile(r"\d+\s?%|\$\s?\d|\b\d+(\.\d+)?\s?(billion|million|trillion)\b", re.I)

# ------------------------------------------------- finance-tuned lexicon ---

# Loughran-McDonald-flavoured word lists for Finnhub articles (no sentiment given)
LM_POSITIVE = {
    "growth", "beat", "beats", "raised", "raise", "upgrade", "upgraded", "record",
    "surge", "surged", "profitable", "expansion", "expanding", "strong", "robust",
    "momentum", "optimistic", "bullish", "dividend", "buyback", "outperform",
    "exceed", "exceeded", "all-time high", "breakthrough", "profit",
}
LM_NEGATIVE = {
    "loss", "losses", "miss", "missed", "cut", "cuts", "downgrade", "downgraded",
    "lawsuit", "fraud", "investigation", "fine", "fined", "bankruptcy", "layoff",
    "layoffs", "decline", "declined", "weak", "slump", "plunge", "plunged",
    "warning", "risk", "volatile", "bearish", "recall", "breach", "default",
    "probe", "scandal", "restructuring", "shortfall",
}


def lexicon_sentiment(text: str) -> float:
    """Finance-tuned lexicon sentiment in [-1, 1]."""
    t = text.lower()
    pos = sum(1 for w in LM_POSITIVE if re.search(r"\b" + re.escape(w) + r"\b", t))
    neg = sum(1 for w in LM_NEGATIVE if re.search(r"\b" + re.escape(w) + r"\b", t))
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg + 1)


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


# ------------------------------------------------------------- NER (opt) ---

_NLP = None
_NLP_TRIED = False


def _get_nlp():
    """spaCy pipeline if available, else None (fail-soft)."""
    global _NLP, _NLP_TRIED
    if not _NLP_TRIED:
        _NLP_TRIED = True
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = None
    return _NLP


_NAME_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc",
    "co", "company", "holdings", "holding", "group", "sa", "ag", "nv", "llc",
    "the",
}


def _name_tokens(company_name: str) -> list[str]:
    toks = re.findall(r"[a-z0-9&]+", company_name.lower())
    return [t for t in toks if t not in _NAME_SUFFIXES and len(t) > 1]


def entity_relevance(article: RawArticle, ticker: str, company_name: str) -> float:
    """0..1 score: is this article actually about the requested company?

    A strong API ticker signal is only trusted when the article text itself
    corroborates it (company name tokens, the ticker string, fuzzy title
    match, or spaCy ORG NER). Otherwise it's a passing mention and is
    downgraded below the relevance gate.
    """
    t = ticker.upper()
    text = f"{article.title} {article.summary}".lower()
    text_tokens = set(re.findall(r"[a-z0-9&]+", text))
    name_toks = _name_tokens(company_name)

    token_hit = bool(name_toks) and any(tok in text_tokens for tok in name_toks)
    ticker_token_hit = t.lower() in text_tokens
    all_tokens_hit = bool(name_toks) and all(tok in text_tokens for tok in name_toks)

    def fuzzy_hit() -> bool:
        if not name_toks:
            return False
        return difflib.SequenceMatcher(None, " ".join(name_toks), article.title.lower()).ratio() >= 0.8

    def ner_hit() -> bool:
        nlp = _get_nlp()
        if nlp is None or not name_toks:
            return False
        try:
            doc = nlp(article.title + ". " + article.summary[:500])
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    r = difflib.SequenceMatcher(None, " ".join(name_toks), ent.text.lower()).ratio()
                    if r >= 0.85:
                        return True
        except Exception:
            pass
        return False

    corroborated = token_hit or ticker_token_hit or fuzzy_hit() or ner_hit()

    if t in (article.tickers or []):
        base = article.ticker_relevance.get(t, 0.9)
        try:
            base = float(base)
        except (TypeError, ValueError):
            base = 0.9
        if base >= 0.85:
            # strong API signal: require textual corroboration
            return 0.90 if corroborated else 0.30
        return base  # trust Alpha Vantage's own weak-relevance judgment

    if corroborated:
        return 0.90 if all_tokens_hit else 0.80
    return 0.15


# ------------------------------------------------- filing-spam filter ---

# 13F / institutional-holdings disclosure spam: backward-looking regulatory
# filings ("Fund X buys 12,000 shares of Apple") that drown real news.
_FILING_RES = [
    re.compile(r"\b13f\b", re.I),
    re.compile(r"\b(buys?|sells?|acquires?|purchases?)\b.{0,50}?\bshares?\b", re.I),
    re.compile(r"\bshares?\b.{0,40}?\b(acquired|purchased|bought|sold)\b.{0,25}?\bby\b", re.I),
    re.compile(r"\bmakes?\b.{0,15}?\bnew\b.{0,40}?\binvestment in\b", re.I),
    re.compile(r"\bnew (position|stake|holdings?)\b", re.I),
    re.compile(r"\b(quarterly|13-f) (filing|holdings?|disclosure)\b", re.I),
    re.compile(r"\binstitutional (investor|ownership|holdings?)\b", re.I),
    re.compile(r"\bhedge fund\b.{0,40}?\b(position|stake|shares)\b", re.I),
    re.compile(r"\b(holdings?|stake|position)\b.{0,30}?\b(increased|decreased|boosted|trimmed|raised|cut)\b", re.I),
    re.compile(r"\b(discloses?|reports?)\b.{0,30}?\b(stake|position|holdings?)\b.{0,20}?\bin\b", re.I),
]


def _is_filing_spam(text: str) -> bool:
    # don't nuke genuine M&A ("acquires Activision for $69B")
    if re.search(r"\bacquires?\b.{0,40}?\bfor\b.{0,20}?\$", text, re.I):
        return False
    return any(rx.search(text) for rx in _FILING_RES)


# ------------------------------------------------- financial relevance ---

def financial_relevance(article: RawArticle) -> tuple[float, list[str]]:
    """0..1 score + category labels: does a PM care about this news?"""
    text = f"{article.title} {article.summary}".lower()

    # 13F-style filing spam is never PM-material, no matter the keywords
    if _is_filing_spam(text):
        return 0.15, ["filings"]

    best = 0.0
    cats: list[str] = []

    for cat, spec in FINANCE_TAXONOMY.items():
        hits = sum(1 for kw in spec["keywords"] if re.search(r"\b" + re.escape(kw) + r"\b", text))
        s = min(1.0, hits / 3)
        if s >= 0.30 and cat not in cats:
            cats.append(cat)
        best = max(best, spec["weight"] * s)

    for topic, rel in article.topics or []:
        cat = AV_TOPIC_MAP.get(topic)
        if cat:
            if cat not in cats:
                cats.append(cat)
            best = max(best, FINANCE_TAXONOMY[cat]["weight"] * rel)

    # numbers near money talk amplify materiality ("opex cut 20%", "$4B buyback")
    if best > 0.1 and _NUMERIC_RE.search(text):
        best = min(1.0, best + 0.15)

    return best, cats or ["general"]


# ------------------------------------------------------------- sources ---

_TIER1 = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "barrons.com", "forbes.com", "economist.com",
    "nytimes.com", "washingtonpost.com", "apnews.com",
}
_TIER2 = {
    "finance.yahoo.com", "yahoo.com", "seekingalpha.com", "benzinga.com",
    "zacks.com", "fool.com", "investopedia.com", "thestreet.com",
    "investing.com", "morningstar.com", "tipranks.com", "gurufocus.com",
}


def source_weight(article: RawArticle) -> float:
    dom = (article.source_domain or "").lower()
    name = (article.source or "").lower()
    hay = f"{dom} {name}"
    if any(d in hay for d in _TIER1):
        return 1.0
    if any(d in hay for d in _TIER2):
        return 0.8
    return 0.6


# ------------------------------------------------------------ pipeline ---

@dataclass
class ScoredArticle:
    article: RawArticle
    relevance: float
    financial_relevance: float
    categories: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    recency: float = 0.0
    materiality: float = 0.0
    priority: bool = False
    age_days: int = 0


def score_article(
    article: RawArticle,
    ticker: str,
    company_name: str,
    days: int,
    priority_days: int,
    now: datetime,
) -> ScoredArticle:
    relevance = entity_relevance(article, ticker, company_name)
    fin_rel, cats = financial_relevance(article)

    # sentiment: prefer Alpha Vantage per-ticker score, else lexicon
    t = ticker.upper()
    if t in (article.ticker_sentiment or {}):
        s_score = article.ticker_sentiment[t]
    elif article.overall_sentiment is not None:
        s_score = article.overall_sentiment
    else:
        s_score = lexicon_sentiment(f"{article.title} {article.summary}")

    age_days = max(0, (now - article.published_at).days)
    if age_days < priority_days:
        recency = 1.0
    else:
        span = max(1, days - priority_days)
        recency = max(0.25, 1.0 - 0.75 * (age_days - priority_days) / span)

    materiality = (
        0.30 * relevance
        + 0.40 * fin_rel
        + 0.20 * recency
        + 0.10 * source_weight(article)
    )
    # Hard gate: financially irrelevant news can never be material, no matter
    # how recent or on-topic ("athlete wins gold" stays out, always).
    if fin_rel < 0.30:
        materiality = min(materiality, 0.44)

    return ScoredArticle(
        article=article,
        relevance=round(relevance, 3),
        financial_relevance=round(fin_rel, 3),
        categories=cats,
        sentiment_score=round(float(s_score), 3),
        sentiment_label=sentiment_label(float(s_score)),
        recency=round(recency, 3),
        materiality=round(materiality, 3),
        priority=age_days < priority_days,
        age_days=age_days,
    )


def preprocess(
    articles: list[RawArticle],
    ticker: str,
    company_name: str,
    days: int,
    priority_days: int,
    materiality_threshold: float = 0.45,
    relevance_threshold: float = 0.35,
) -> tuple[list[ScoredArticle], list[ScoredArticle]]:
    """Return (relevant, material): relevant passes entity gate, material passes all."""
    now = datetime.now(timezone.utc)
    scored = [score_article(a, ticker, company_name, days, priority_days, now) for a in articles]
    relevant = [s for s in scored if s.relevance >= relevance_threshold]
    material = sorted(
        [s for s in relevant if s.materiality >= materiality_threshold],
        key=lambda s: s.materiality,
        reverse=True,
    )
    return relevant, material
