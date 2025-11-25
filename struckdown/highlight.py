"""Syntax highlighting for struckdown files."""

import re
from html import escape


# patterns ordered by priority (longer/more specific first)
# each tuple: (class_name, pattern, flags)
# NOTE: system tags are handled separately in highlight_struckdown_with_system_blocks
PATTERNS = [
    # comments first (can contain other syntax)
    ('comment', r'<!--[\s\S]*?-->', 0),
    # section separators (must be at line start)
    ('section-sep', r'^---#[a-zA-Z_][a-zA-Z0-9_]*$', re.MULTILINE),
    # XML-style tags (system tags handled separately)
    ('checkpoint', r'<checkpoint[^>]*/?\s*>', 0),
    ('obliviate', r'<obliviate[^>]*/?\s*>', 0),
    ('include', r'<include[^>]*/>', 0),
    ('break-open', r'<break[^>]*>', 0),
    ('break-close', r'</break>', 0),
    # placeholders and template syntax
    ('placeholder', r'\[\[[^\]]+\]\]', 0),
    ('template-var', r'\{\{[^}]+\}\}', 0),
    ('jinja-tag', r'\{%[^%]+%\}', 0),
]


def highlight_struckdown_with_system_blocks(text: str) -> str:
    """Apply syntax highlighting with special handling for system blocks.

    System blocks get a light background, with darker backgrounds on tag lines.
    """
    # first, identify system block regions
    system_block_pattern = re.compile(r'(<system[^>]*>)([\s\S]*?)(</system>)')

    result = []
    last_end = 0

    for match in system_block_pattern.finditer(text):
        # add text before this system block (with normal highlighting)
        if match.start() > last_end:
            result.append(highlight_struckdown(text[last_end:match.start()]))

        open_tag = match.group(1)
        content = match.group(2)
        close_tag = match.group(3)

        # opening tag line - darker background, bold black text
        result.append(f'<span class="sd-system-tag-line"><span class="sd-system-tag">{escape(open_tag)}</span></span>')

        # content - lighter background, with inner highlighting
        if content.strip():
            # highlight placeholders and vars within system content
            inner_highlighted = highlight_struckdown(content)
            result.append(f'<span class="sd-system-content">{inner_highlighted}</span>')
        elif content:
            result.append(f'<span class="sd-system-content">{escape(content)}</span>')

        # closing tag line - darker background, bold black text
        result.append(f'<span class="sd-system-tag-line"><span class="sd-system-tag">{escape(close_tag)}</span></span>')

        last_end = match.end()

    # add remaining text
    if last_end < len(text):
        result.append(highlight_struckdown(text[last_end:]))

    # if no system blocks found, just do normal highlighting
    if last_end == 0:
        return highlight_struckdown(text)

    return ''.join(result)


def highlight_struckdown(text: str) -> str:
    """Apply syntax highlighting to struckdown text.

    Returns HTML with spans for each syntax element.
    The text is HTML-escaped first, then highlighting is applied.
    """
    # build combined pattern with named groups
    combined_parts = []
    for name, pattern, flags in PATTERNS:
        # replace hyphens with underscores for valid group names
        group_name = name.replace('-', '_')
        combined_parts.append(f'(?P<{group_name}>{pattern})')

    combined_pattern = '|'.join(combined_parts)

    # we need to track positions and build result
    result = []
    last_end = 0

    for match in re.finditer(combined_pattern, text, re.MULTILINE):
        # add text before match (escaped)
        if match.start() > last_end:
            result.append(escape(text[last_end:match.start()]))

        # find which group matched
        for name, _, _ in PATTERNS:
            group_name = name.replace('-', '_')
            matched_text = match.group(group_name)
            if matched_text is not None:
                # wrap in span with class
                escaped_match = escape(matched_text)
                result.append(f'<span class="sd-{name}">{escaped_match}</span>')
                break

        last_end = match.end()

    # add remaining text
    if last_end < len(text):
        result.append(escape(text[last_end:]))

    return ''.join(result)


def render_preview_html(text: str, filename: str = "preview") -> str:
    """Render a complete HTML preview page with syntax highlighting."""
    highlighted = highlight_struckdown_with_system_blocks(text)

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{escape(filename)} - Struckdown Preview</title>
    <style>
        @font-face {{
            font-family: 'Inconsolata';
            font-style: normal;
            font-weight: 400 700;
            src: url(https://fonts.gstatic.com/s/inconsolata/v32/QldgNThLqRwH-OJ1UHjlKENVzkWGVkL3GZQmAwLYxYWI2qfdm7Lpp4U8WR32lw.woff2) format('woff2');
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Inconsolata', ui-monospace, "SF Mono", "Cascadia Code", monospace;
            background: #fafafa;
            padding: 2em;
            max-width: 900px;
            margin: 0 auto;
            color: #333;
        }}
        h1 {{
            font-size: 1.2em;
            color: #666;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.5em;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            background: white;
            padding: 1.5em;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            font-size: 16px;
            overflow-x: auto;
        }}
        /* placeholder [[...]] - emerald */
        .sd-placeholder {{
            background: rgba(6, 214, 160, 0.3);
            color: #047857;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid rgba(6, 214, 160, 0.6);
        }}
        /* template variables {{...}} - bubblegum pink */
        .sd-template-var {{
            background: rgba(239, 71, 111, 0.25);
            color: #be3a5a;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid rgba(239, 71, 111, 0.5);
        }}
        /* jinja tags {{% %}} - ocean blue, no bg */
        .sd-jinja-tag {{
            color: #118ab2;
        }}
        /* system tag line - grey */
        .sd-system-tag-line {{
            background: #e0e0e0;
            display: block;
            padding: 4px 12px;
            margin: 0 -1.5em;
            padding-left: 1.5em;
            padding-right: 1.5em;
            border-left: 4px solid #9e9e9e;
        }}
        /* system tag text - bold dark */
        .sd-system-tag {{
            font-weight: bold;
            color: #424242;
        }}
        /* system content - light grey */
        .sd-system-content {{
            background: #f5f5f5;
            display: block;
            padding: 8px 12px;
            margin: 0 -1.5em;
            padding-left: 1.5em;
            padding-right: 1.5em;
            border-left: 4px solid #bdbdbd;
        }}
        /* checkpoint - golden pollen */
        .sd-checkpoint {{
            background: #ffd166;
            color: #073b4c;
            font-weight: bold;
            display: block;
            padding: 6px 8px;
            margin: 0.5em -1.5em;
            padding-left: 1.5em;
            padding-right: 1.5em;
        }}
        /* obliviate - bubblegum pink */
        .sd-obliviate {{
            background: #ef476f;
            color: #ffffff;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 8px;
        }}
        /* include - ocean blue */
        .sd-include {{
            background: rgba(17, 138, 178, 0.2);
            color: #0d6e8a;
            font-weight: 500;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid rgba(17, 138, 178, 0.5);
        }}
        /* break - bubblegum pink warning */
        .sd-break-open, .sd-break-close {{
            background: #ef476f;
            color: #ffffff;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 8px;
        }}
        /* comments - muted */
        .sd-comment {{
            color: #888888;
            font-style: italic;
            opacity: 0.8;
        }}
        /* section separators - dark teal banner */
        .sd-section-sep {{
            color: #ffffff;
            font-weight: bold;
            display: block;
            margin: 0.5em 0;
            background: #073b4c;
            padding: 4px 12px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>{escape(filename)}</h1>
    <pre>{highlighted}</pre>
</body>
</html>'''
