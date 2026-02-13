"""
URL fetching with HTTP caching using requests-cache.

Uses the same cache directory as the main struckdown cache (STRUCKDOWN_CACHE env var).
Handles ETag/Last-Modified/Cache-Control headers automatically.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests_cache

from .cache import get_cache_dir

logger = logging.getLogger(__name__)

_cached_session: Optional[requests_cache.CachedSession] = None


def _get_cached_session() -> requests_cache.CachedSession:
    """Get or create a cached requests session."""
    global _cached_session
    if _cached_session is not None:
        return _cached_session

    cache_dir = get_cache_dir()

    if cache_dir is None:
        # caching disabled -- use a session that doesn't cache
        _cached_session = requests_cache.CachedSession(
            backend="memory",
            expire_after=0,  # immediately expire = no caching
        )
        logger.info("URL caching disabled")
    else:
        url_cache_dir = cache_dir / "url_cache"
        url_cache_dir.mkdir(parents=True, exist_ok=True)

        _cached_session = requests_cache.CachedSession(
            cache_name=str(url_cache_dir / "http_cache"),
            backend="sqlite",
            expire_after=3600,  # 1 hour default, but ETag/Last-Modified will revalidate
            stale_if_error=True,  # use stale cache if server errors
            cache_control=True,  # respect Cache-Control headers
        )
        logger.info(f"URL cache initialised at {url_cache_dir}")

    return _cached_session


def is_url(s: str) -> bool:
    """Check if string is a URL."""
    return s.startswith("http://") or s.startswith("https://")


def fetch_url(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Fetch URL content with HTTP caching.

    Returns (content, content_type) tuple.
    Caching is handled automatically via ETag/Last-Modified/Cache-Control.
    """
    session = _get_cached_session()

    headers = {"User-Agent": "struckdown/1.0"}
    response = session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    content = response.text
    content_type = response.headers.get("Content-Type", "").lower()

    # log cache status
    if hasattr(response, "from_cache") and response.from_cache:
        logger.info(f"URL fetched from cache: {url}")
    else:
        logger.info(f"URL fetched from network: {url}")

    return content, content_type


def read_input_url(url: str) -> List[dict]:
    """
    Fetch and parse input from a URL.

    Supports JSON and CSV based on Content-Type header or URL extension.
    Uses HTTP caching with ETag/Last-Modified for efficiency.
    """
    import io

    import pandas as pd

    logger.info(f"Fetching input from URL: {url}")

    content, content_type = fetch_url(url)

    # determine format from content-type or URL extension
    parsed = urlparse(url)
    path_lower = parsed.path.lower()

    if "application/json" in content_type or path_lower.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "filename" not in item:
                    item["filename"] = url
            return data
        elif isinstance(data, dict):
            if "filename" not in data:
                data["filename"] = url
            return [data]
        else:
            raise ValueError(f"JSON must be dict or list, got {type(data)}")

    elif "text/csv" in content_type or path_lower.endswith(".csv"):
        df = pd.read_csv(io.StringIO(content), keep_default_na=False, na_values=[""])
        original_columns = list(df.columns)
        rows = df.where(pd.notnull(df), None).to_dict(orient="records")
        result = []
        for row_data in rows:
            result.append({
                "_original_columns": original_columns,
                **row_data,
            })
        logger.info(f"Loaded {len(rows)} rows from URL with columns: {original_columns}")
        return result

    else:
        # treat as plain text
        return [{
            "input": content,
            "content": content,
            "source": content,
            "filename": url,
        }]
