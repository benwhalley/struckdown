"""Built-in @fetch action for fetching URLs."""

from . import Actions, fetch_and_parse, is_url, DEFAULT_MAX_CHARS, DEFAULT_TIMEOUT


@Actions.register("fetch", on_error="propagate", default_save=True, allow_remote_use=False)
def fetch_action(
    context: dict,
    url: str = "",
    raw: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS,
    playwright: bool = False,
) -> str:
    """Fetch URL content and return as markdown (or raw HTML).

    Usage:
        [[@fetch:content|{{product_url}}]]
        [[@fetch:raw_html|{{url}},raw=true]]
        [[@fetch:page|{{url}},max_chars=0]]
        [[@fetch:page|{{url}},playwright=true]]

    Args:
        context: Struckdown context dict
        url: The URL to fetch (can be a template variable)
        raw: If true, return raw HTML instead of cleaned markdown
        timeout: Request timeout in seconds (default 30)
        max_chars: Maximum characters to return (default 32000, 0 = no limit)
        playwright: If true, use Playwright browser; otherwise uses requests
            with automatic fallback to Playwright on 403/401 errors.
            Requires: pip install struckdown[playwright] && playwright install chromium

    Returns:
        Content string (raw HTML or cleaned markdown)
    """
    url = str(url).strip()
    if not url:
        raise ValueError("URL is required for @fetch action")

    if not is_url(url):
        raise ValueError(f"Invalid URL: {url}")

    return fetch_and_parse(url, raw=raw, timeout=timeout, max_chars=max_chars, playwright=playwright)
