import asyncio
import random
from xml.etree import ElementTree

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    HEADERS,
    MAX_CONCURRENT,
    MAX_RETRIES,
    REQUEST_DELAY_MAX,
    REQUEST_DELAY_MIN,
    REQUEST_TIMEOUT,
    SITEMAP_URL,
)

SKIP_PATTERNS = ["/exercises", "/quizzes", "/login", "/tryit", "/game", "/quiz"]


async def fetch_sitemap_urls() -> list[str]:
    async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(SITEMAP_URL)
        resp.raise_for_status()

    root = ElementTree.fromstring(resp.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]

    return [u for u in urls if not any(p in u for p in SKIP_PATTERNS)]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
)
async def fetch_page(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    resp = await client.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return url, resp.text


async def fetch_all(urls: list[str]) -> list[tuple[str, str]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results: list[tuple[str, str]] = []

    async def bounded_fetch(client: httpx.AsyncClient, url: str):
        async with semaphore:
            await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
            try:
                return await fetch_page(client, url)
            except Exception as exc:
                print(f"[SKIP] {url}: {exc}")
                return None

    async with httpx.AsyncClient(headers=HEADERS, http2=True) as client:
        tasks = [bounded_fetch(client, url) for url in urls]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)

    return results
