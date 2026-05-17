import asyncio
import random
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    BASE_URL,
    HEADERS,
    MAX_CONCURRENT,
    MAX_RETRIES,
    REQUEST_DELAY_MAX,
    REQUEST_DELAY_MIN,
    REQUEST_TIMEOUT,
)

SKIP_PATTERNS = ["/exercises", "/quizzes", "/login", "/tryit", "/game", "/quiz", "/w3css/"]

# Topic index pages to crawl for tutorial links
TOPIC_INDEXES = [
    "/html/", "/css/", "/js/", "/python/", "/sql/",
    "/php/", "/java/", "/c/", "/cpp/", "/csharp/",
    "/jquery/", "/bootstrap/", "/react/", "/nodejs/",
    "/typescript/", "/xml/", "/git/", "/r/", "/kotlin/",
]


async def _discover_links_from_index(client: httpx.AsyncClient, index_path: str) -> list[str]:
    """Fetch a topic index page and return all linked tutorial URLs."""
    base_page_url = urljoin(BASE_URL, index_path)
    try:
        resp = await client.get(base_page_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    links = set()
    topic = index_path.strip("/")

    for a in soup.find_all("a", href=True):
        href: str = a["href"]

        # Resolve relative and absolute hrefs against the index page URL
        full = urljoin(base_page_url, href)

        # Keep only pages within this topic, ending in .asp or .php
        if (
            f"/{topic}/" in full
            and full.startswith(BASE_URL)
            and (full.endswith(".asp") or full.endswith(".php"))
            and not any(p in full for p in SKIP_PATTERNS)
        ):
            links.add(full)

    return list(links)


async def fetch_sitemap_urls() -> list[str]:
    """Discover all tutorial URLs by crawling topic index pages."""
    all_urls: set[str] = set()

    async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as client:
        tasks = [_discover_links_from_index(client, idx) for idx in TOPIC_INDEXES]
        results = await asyncio.gather(*tasks)

    for url_list in results:
        all_urls.update(url_list)

    return sorted(all_urls)


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
