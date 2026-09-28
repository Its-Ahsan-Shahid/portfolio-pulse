# Portfolio Pulse — single-container deploy (Hugging Face Spaces Docker / any host)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCy NER model (optional at runtime, but baked in for entity relevance)
RUN python -m spacy download en_core_web_sm

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/ ./backend/

EXPOSE 7860

# HF Spaces Docker expects the app on 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
