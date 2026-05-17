"""
One-time migration: local ChromaDB → Qdrant Cloud.
Run after the scraper pipeline finishes:
    PYTHONPATH=. .venv/bin/python scraper/migrate_to_qdrant.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from config import CHROMA_PATH, COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

VECTOR_SIZE = 384
BATCH_SIZE = 500


def migrate():
    print("=== ChromaDB → Qdrant Migration ===\n")

    # --- Source: local ChromaDB ---
    chroma = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    chroma_col = chroma.get_collection(COLLECTION_NAME)
    total = chroma_col.count()
    print(f"Source ChromaDB: {total:,} chunks at {CHROMA_PATH}")

    # --- Destination: Qdrant Cloud ---
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION_NAME in existing:
        print(f"Deleting existing Qdrant collection '{COLLECTION_NAME}'...")
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection '{COLLECTION_NAME}' (cosine, {VECTOR_SIZE}d)\n")

    # --- Batch migrate ---
    offset = 0
    migrated = 0
    with tqdm(total=total, desc="Migrating") as pbar:
        while offset < total:
            batch = chroma_col.get(
                limit=BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )
            if not batch["ids"]:
                break

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": doc, **meta},
                )
                for doc, meta, embedding in zip(
                    batch["documents"], batch["metadatas"], batch["embeddings"]
                )
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            n = len(batch["ids"])
            offset += n
            migrated += n
            pbar.update(n)

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\nDone! {info.points_count:,} points in Qdrant collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: Set QDRANT_URL and QDRANT_API_KEY in your .env file first.")
        sys.exit(1)
    migrate()
