from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

VECTOR_SIZE = 384  # BAAI/bge-small-en-v1.5 output dimension

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        existing = {c.name for c in _client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        _client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="topic",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    return _client


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                "url": chunk["url"],
                "title": chunk["title"],
                "topic": chunk["topic"],
                "heading": chunk["heading"],
                "has_code": chunk["has_code"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def query(
    query_embedding: list[float],
    top_k: int = 10,
    topic_filter: str | None = None,
) -> list[dict]:
    client = get_client()
    query_filter = (
        Filter(must=[FieldCondition(key="topic", match=MatchValue(value=topic_filter))])
        if topic_filter
        else None
    )
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        {
            "text": r.payload["text"],
            "metadata": {
                "url": r.payload["url"],
                "title": r.payload["title"],
                "topic": r.payload["topic"],
                "heading": r.payload.get("heading", ""),
            },
            "score": r.score,
        }
        for r in response.points
    ]


def collection_stats() -> dict:
    info = get_client().get_collection(COLLECTION_NAME)
    return {"total_chunks": info.points_count}
