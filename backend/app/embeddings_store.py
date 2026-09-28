"""Vector store: ChromaDB (primary) with a TF-IDF fallback.

An ephemeral collection is created per analysis job and deleted afterwards,
so there is no cross-user data leakage. If ChromaDB / sentence-transformers
are unavailable (e.g. constrained environment), we fall back to an in-memory
TF-IDF + cosine similarity store with the same interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DocHit:
    text: str
    metadata: dict
    score: float


class BaseStore:
    def add(self, docs: list[dict]) -> None:
        raise NotImplementedError

    def query(self, text: str, n: int = 10) -> list[DocHit]:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ------------------------------------------------------------- ChromaDB ---

class _STEmbeddingFunction:
    """ChromaDB-compatible embedding function wrapping sentence-transformers."""

    def __init__(self, model):
        self._model = model

    def __call__(self, input):  # chroma passes `input` kwarg
        embs = self._model.encode(list(input), normalize_embeddings=True)
        return [list(map(float, e)) for e in embs]


class ChromaStore(BaseStore):
    def __init__(self, persist_dir: str, collection_name: str):
        import chromadb
        from sentence_transformers import SentenceTransformer

        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", collection_name)[:60]
        if not safe[0].isalnum():
            safe = "c" + safe[1:]
        if not safe[-1].isalnum():
            safe = safe[:-1] + "c"
        self._name = safe or "pp-default"

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._client = chromadb.PersistentClient(path=persist_dir)
        # fresh collection per job
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            self._name, embedding_function=_STEmbeddingFunction(model)
        )

    def add(self, docs: list[dict]) -> None:
        if not docs:
            return
        self._col.add(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d.get("metadata", {}) for d in docs],
        )

    def query(self, text: str, n: int = 10) -> list[DocHit]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_texts=[text], n_results=min(n, self._col.count()))
        hits: list[DocHit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(DocHit(text=doc, metadata=meta or {}, score=float(1.0 - (dist or 0))))
        return hits

    def close(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass


# ---------------------------------------------------------------- TF-IDF ---

class TfidfStore(BaseStore):
    """Zero-dependency-robust fallback: sklearn TF-IDF + cosine similarity."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(stop_words="english", max_features=5000)
        self._docs: list[dict] = []
        self._matrix = None

    def add(self, docs: list[dict]) -> None:
        import numpy as np  # noqa: F401  (ensures numpy present)

        self._docs = docs
        if docs:
            self._matrix = self._vec.fit_transform([d["text"] for d in docs])

    def query(self, text: str, n: int = 10) -> list[DocHit]:
        import numpy as np

        if not self._docs or self._matrix is None:
            return []
        q = self._vec.transform([text])
        scores = (self._matrix @ q.T).toarray().ravel()
        norms = np.sqrt((self._matrix.multiply(self._matrix)).sum(axis=1)).A.ravel()
        qnorm = float(np.sqrt((q.multiply(q)).sum())) or 1e-9
        cos = scores / (norms * qnorm + 1e-9)
        idx = np.argsort(-cos)[:n]
        return [
            DocHit(text=self._docs[i]["text"], metadata=self._docs[i].get("metadata", {}), score=float(cos[i]))
            for i in idx
            if cos[i] > 0
        ]


def build_store(persist_dir: str, collection_name: str) -> tuple[BaseStore, str]:
    """Return (store, backend_name). Falls back gracefully, always works."""
    try:
        return ChromaStore(persist_dir, collection_name), "chromadb"
    except Exception as e:
        print(f"[portfolio-pulse] ChromaDB unavailable ({e}); trying TF-IDF")
    try:
        return TfidfStore(), "tfidf"
    except Exception as e:
        print(f"[portfolio-pulse] TF-IDF unavailable ({e}); using keyword overlap")
        return KeywordStore(), "keyword"


class KeywordStore(BaseStore):
    """Pure-stdlib fallback: token-overlap scoring. No numpy/sklearn needed."""

    def __init__(self):
        self._docs: list[dict] = []

    def add(self, docs: list[dict]) -> None:
        self._docs = docs

    def query(self, text: str, n: int = 10) -> list[DocHit]:
        import re as _re

        qtokens = set(_re.findall(r"[a-z0-9]+", text.lower()))
        scored = []
        for d in self._docs:
            dtokens = set(_re.findall(r"[a-z0-9]+", d["text"].lower()))
            overlap = len(qtokens & dtokens)
            if overlap:
                scored.append((overlap / max(1, len(qtokens)), d))
        scored.sort(key=lambda x: -x[0])
        return [
            DocHit(text=d["text"], metadata=d.get("metadata", {}), score=float(s))
            for s, d in scored[:n]
        ]


def doc_text_for(scored) -> str:
    a = scored.article
    return (
        f"{a.title}\n{a.summary}\n"
        f"Categories: {', '.join(scored.categories)} | "
        f"Sentiment: {scored.sentiment_label} ({scored.sentiment_score}) | "
        f"Materiality: {scored.materiality} | Source: {a.source}"
    )
