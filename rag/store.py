from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings

from config import CHROMA_PATH, COLLECTION_NAME

_client: chromadb.PersistentClient | None = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection = get_collection()
    collection.upsert(
        ids=[str(uuid.uuid4()) for _ in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "url": c["url"],
                "title": c["title"],
                "topic": c["topic"],
                "heading": c["heading"],
                "has_code": str(c["has_code"]),
            }
            for c in chunks
        ],
    )


def query(
    query_embedding: list[float],
    top_k: int = 10,
    topic_filter: str | None = None,
) -> list[dict]:
    collection = get_collection()
    where = {"topic": topic_filter} if topic_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": 1 - dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_stats() -> dict:
    return {"total_chunks": get_collection().count()}
