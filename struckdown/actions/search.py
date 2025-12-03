"""Built-in @search action for web searching via DuckDuckGo."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

from ddgs import DDGS

from . import (
    Actions,
    fetch_url,
    extract_readable,
    html_to_markdown,
    truncate_content,
    DEFAULT_TIMEOUT,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 10000


@dataclass
class SearchResult:
    """A single search result with fetched content."""
    title: str
    url: str
    snippet: str
    content: str = ""

    def __str__(self) -> str:
        """Format as readable text."""
        parts = [f"## {self.title}", f"URL: {self.url}", f"Snippet: {self.snippet}"]
        if self.content:
            parts.append(f"\n### Content:\n{self.content}")
        return "\n".join(parts)


def fetch_page_content(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    raw: bool = False,
    playwright: bool = False,
) -> str:
    """Fetch and parse a single page.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_tokens: Max characters to return (0 = no limit)
        raw: If True, return raw content; if False and HTML, markdownify
        playwright: If True, use Playwright browser; if False, use requests
            with automatic fallback to Playwright on 403/401 errors

    Returns:
        Page content (raw, or markdown if HTML)
    """
    if not url or not url.strip():
        return ""
    try:
        content, ctype = fetch_url(url, timeout=timeout, playwright=playwright)
        # only process as HTML if content-type indicates HTML and raw=False
        if raw or "html" not in ctype:
            return truncate_content(content, max_tokens)
        readable = extract_readable(content)
        md = html_to_markdown(readable)
        return truncate_content(md, max_tokens)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return ""


def search_and_fetch(
    query: str,
    max_results: int = 5,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    raw: bool = False,
    playwright: bool = False,
) -> List[SearchResult]:
    """Search DuckDuckGo and fetch page content in parallel.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        timeout: Request timeout in seconds
        max_tokens: Max chars per fetched page (0 = no limit)
        raw: If True, return raw HTML; if False, markdownify
        playwright: If True, use Playwright browser; if False, use requests
            with automatic fallback to Playwright on 403/401 errors

    Returns:
        List of SearchResult objects with fetched content
    """
    # search using ddgs (avoid context manager due to curl_cffi malloc bug on macOS)
    ddgs = DDGS()
    raw_results = list(ddgs.text(query, max_results=max_results))

    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
        )
        for r in raw_results
        if r.get("href")  # filter out results without URLs
    ]

    if not results:
        return results

    # log URLs being fetched
    logger.info(f"Fetching {len(results)} pages:")
    for r in results:
        logger.info(f"  - {r.url}")

    # fetch all pages in parallel
    with ThreadPoolExecutor(max_workers=min(len(results), 10)) as executor:
        future_to_idx = {
            executor.submit(
                fetch_page_content, result.url, timeout, max_tokens, raw, playwright
            ): idx
            for idx, result in enumerate(results)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx].content = future.result()
            except Exception as e:
                logger.warning(f"Error fetching result {idx}: {e}")

    return results


@Actions.register("search", allow_remote_use=False)
def search_action(
    context: dict,
    query: str,
    max_results: int = 5,
    embed: bool = True,
    raw: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    playwright: bool = False,
) -> str:
    """Search DuckDuckGo and return results with fetched page content.

    By default, fetches each result page, converts to markdown, and embeds
    the full content. Use embed=false to just store results for later use
    via {{varname}}.

    Usage:
        [[@search:info|{{planet}}]]                    # embed full content
        [[@search:info|{{topic}},embed=false]]         # store for {{info}}
        [[@search:info|{{topic}},raw=true]]            # don't markdownify
        [[@search:info|{{topic}},max_tokens=0]]        # no truncation
        [[@search:results|query="news",max_results=3]]
        [[@search:info|{{topic}},playwright=true]]     # use headless browser

    Args:
        context: Struckdown context dict
        query: Search query string
        max_results: Maximum number of results to return (default 5)
        embed: If True, embed full page content; if False, just titles/snippets
        raw: If True, don't markdownify fetched pages (return raw HTML)
        timeout: Request timeout in seconds (default 30)
        max_tokens: Max characters per fetched page (default 10000, 0 = no limit)
        playwright: If True, use Playwright browser for fetching; if False, use
            requests with automatic fallback to Playwright on 403/401 errors.
            Requires: pip install struckdown[playwright] && playwright install chromium

    Returns:
        Formatted search results (full content if embed=True, summary if embed=False)
    """
    logger.info(f"Searching for: {query}")

    results = search_and_fetch(
        query=query,
        max_results=max_results,
        timeout=timeout,
        max_tokens=max_tokens,
        raw=raw,
        playwright=playwright,
    )

    if not results:
        return f"No results found for: {query}"

    if embed:
        # full content embedded
        formatted = [f"# Search Results for: {query}\n"]
        for i, result in enumerate(results, 1):
            formatted.append(f"## Result {i}: {result.title}")
            formatted.append(f"URL: {result.url}")
            if result.content:
                formatted.append(f"\n{result.content}\n")
            else:
                formatted.append(f"Snippet: {result.snippet}\n")
            formatted.append("---\n")
        return "\n".join(formatted)
    else:
        # summary only - user can access full content via {{varname}}
        formatted = [f"# Search Results for: {query}\n"]
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. **{result.title}**")
            formatted.append(f"   {result.url}")
            formatted.append(f"   {result.snippet}\n")
        return "\n".join(formatted)
