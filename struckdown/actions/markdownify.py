"""Built-in @markdownify action for converting HTML to markdown."""

from . import Actions, html_to_markdown, extract_readable


@Actions.register("markdownify")
def markdownify_action(
    context: dict,
    html: str,
    extract_content: bool = True,
) -> str:
    """Convert HTML to markdown.

    Usage:
        [[@markdownify:md|html={{raw_html}}]]
        [[@markdownify:md|html={{raw_html}},extract_content=false]]

    Args:
        context: Struckdown context dict
        html: HTML string to convert
        extract_content: If True (default), use readability to extract main content first.
                        If False, convert the full HTML as-is.

    Returns:
        Markdown string
    """
    if extract_content:
        html = extract_readable(html)
    return html_to_markdown(html)
