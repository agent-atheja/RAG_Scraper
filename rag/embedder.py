from __future__ import annotations

from sentence_transformers import SentenceTransformer

from config import EMBED_BATCH_SIZE, EMBED_MODEL

_model: SentenceTransformer | None = None

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {EMBED_MODEL} (first run downloads ~130MB)")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_chunks(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_model()
    embedding = model.encode(_QUERY_PREFIX + query, normalize_embeddings=True)
    return embedding.tolist()
