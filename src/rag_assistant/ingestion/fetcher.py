from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)

_STRIP_TAGS = ["nav", "footer", "header", "script", "style", "aside", ".toc"]
_REQUEST_HEADERS = {
    "User-Agent": "k8s-rag-assistant/0.1 (educational; github.com/vkrishnabhatk/k8s-rag-assistant)"
}


@dataclass
class FetchedDoc:
    url: str
    title: str
    raw_text: str
    fetched_at: datetime
    content_hash: str


def _extract_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "header", "script", "style", "aside"]):
        tag.decompose()
    for el in soup.select(".toc, .td-sidebar, .td-toc, [class*='sidebar']"):
        el.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    main = soup.find("main") or soup.find("article") or soup.find("body")
    text = (
        main.get_text(separator=" ", strip=True)
        if main
        else soup.get_text(separator=" ", strip=True)
    )
    return title, text


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
    retries: int = 3,
) -> FetchedDoc | None:
    for attempt in range(1, retries + 1):
        try:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            title, text = _extract_text(resp.text, url)
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            log.info("fetched", url=url, chars=len(text))
            return FetchedDoc(
                url=url,
                title=title,
                raw_text=text,
                fetched_at=datetime.now(tz=UTC),
                content_hash=content_hash,
            )
        except httpx.HTTPStatusError as exc:
            log.warning("http_error", url=url, status=exc.response.status_code, attempt=attempt)
        except httpx.RequestError as exc:
            log.warning("request_error", url=url, error=str(exc), attempt=attempt)
        if attempt < retries:
            await asyncio.sleep(2**attempt)
    log.error("fetch_failed", url=url)
    return None


async def fetch_docs(urls: list[str], timeout_s: float = 30.0) -> list[FetchedDoc]:
    """Concurrently fetch all URLs and return successfully fetched documents."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        headers=_REQUEST_HEADERS,
    ) as client:
        tasks = [_fetch_one(client, url) for url in urls]
        results = await asyncio.gather(*tasks)

    docs = [r for r in results if r is not None]
    log.info("fetch_complete", total=len(urls), succeeded=len(docs), failed=len(urls) - len(docs))
    if len(docs) < len(urls):
        failed = [u for u, r in zip(urls, results, strict=False) if r is None]
        raise RuntimeError(f"Failed to fetch {len(failed)} URLs: {failed}")
    return docs
