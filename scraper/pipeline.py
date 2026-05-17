import asyncio
import sys

from tqdm import tqdm

from scraper.scraper import fetch_all, fetch_sitemap_urls
from scraper.parser import parse_page
from scraper.chunker import chunk_page
from rag.embedder import embed_chunks
from rag.store import collection_stats, upsert_chunks

UPSERT_BATCH_SIZE = 500


async def run_pipeline(limit: int | None = None) -> None:
    print("=== W3Schools → ChromaDB Pipeline ===\n")

    print("[1/4] Fetching sitemap URLs...")
    urls = await fetch_sitemap_urls()
    if limit:
        urls = urls[:limit]
    print(f"      Found {len(urls)} URLs to scrape\n")

    print("[2/4] Scraping pages (async, rate-limited)...")
    pages_raw = await fetch_all(urls)
    print(f"      Fetched {len(pages_raw)} pages\n")

    print("[3/4] Parsing and chunking...")
    all_chunks: list[dict] = []
    for url, html in tqdm(pages_raw, desc="Parsing"):
        page = parse_page(url, html)
        if page and page.raw_text:
            all_chunks.extend(chunk_page(page))
    print(f"      Generated {len(all_chunks)} chunks\n")

    print("[4/4] Embedding and storing in ChromaDB...")
    for i in tqdm(range(0, len(all_chunks), UPSERT_BATCH_SIZE), desc="Embedding+Storing"):
        batch = all_chunks[i: i + UPSERT_BATCH_SIZE]
        embeddings = embed_chunks([c["text"] for c in batch])
        upsert_chunks(batch, embeddings)

    stats = collection_stats()
    print(f"\n=== Done! {stats['total_chunks']} chunks stored in ChromaDB ===")


if __name__ == "__main__":
    # Pass an integer argument to limit pages for testing: python -m scraper.pipeline 10
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(run_pipeline(limit=limit_arg))
