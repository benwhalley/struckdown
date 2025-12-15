"""Syntax highlighting for struckdown files using the Lark parser.

This module uses the same grammar.lark as the main parser, ensuring highlighting
stays in sync with actual syntax. No manual regex patterns needed.

FUTURE: LSP Server Plan
------------------------
For VS Code integration, implement a Language Server Protocol server:

1. Create struckdown_lsp.py using pygls (Python LSP library)
2. Implement textDocument/semanticTokens using this highlighting logic
3. Implement textDocument/publishDiagnostics for parse errors
4. Add completion provider for known types (pick, bool, think, etc.)
5. Update VS Code extension to be an LSP client instead of using tmLanguage

This would provide:
- Semantic highlighting that matches parser exactly
- Real-time error squiggles
- Autocompletion for completion types
- Hover info for syntax elements

See: https://pygls.readthedocs.io/
"""

import re
from html import escape
from importlib.resources import files
from pathlib import Path

from lark import Lark, Token, Tree


# load grammar
try:
    _grammar_text = (files(__package__) / "grammar.lark").read_text(encoding="utf-8")
except Exception:
    _grammar_path = Path(__file__).parent / "grammar.lark"
    _grammar_text = _grammar_path.read_text(encoding="utf-8")


# parser for highlighting - no transformer, keep raw tree with positions
_highlight_parser = Lark(
    _grammar_text,
    parser="lalr",
    propagate_positions=True,
)


# map grammar rules/tokens to CSS class names
TOKEN_CLASS_MAP = {
    # comments
    'block_comment': 'comment',
    'BLOCK_COMMENT_START': 'comment',
    'BLOCK_COMMENT_CONTENT': 'comment',
    'BLOCK_COMMENT_END': 'comment',
    # checkpoints
    'checkpoint_closed': 'checkpoint',
    'checkpoint_auto': 'checkpoint',
    'CHECKPOINT_CLOSED': 'checkpoint',
    'CHECKPOINT_OPEN': 'checkpoint',
    # obliviate (checkpoint alias)
    'obliviate_closed': 'obliviate',
    'obliviate_auto': 'obliviate',
    'OBLIVIATE_CLOSED': 'obliviate',
    'OBLIVIATE_OPEN': 'obliviate',
    # system tags - handled specially for nested structure
    'system_tag': 'system',
    'SYSTEM_OPEN': 'system-tag',
    'SYSTEM_CLOSE': 'system-tag',
    'SYSTEM_CONTENT': 'system-content',
    # header tags - similar to system but with different styling
    'header_tag': 'header',
    'HEADER_OPEN': 'header-tag',
    'HEADER_CLOSE': 'header-tag',
    'HEADER_CONTENT': 'header-content',
    # include
    'include_tag': 'include',
    'INCLUDE_SRC': 'include',
    # completions [[...]]
    'single_completion': 'placeholder',
    # template variables {{...}}
    'placeholder': 'template-var',
    # jinja tags {% ... %}
    'templatetag': 'jinja-tag',
}


class HighlightSpan:
    """A span of text to highlight."""
    __slots__ = ('start', 'end', 'css_class', 'text')

    def __init__(self, start: int, end: int, css_class: str, text: str):
        self.start = start
        self.end = end
        self.css_class = css_class
        self.text = text

    def __repr__(self):
        return f"HighlightSpan({self.start}, {self.end}, {self.css_class!r})"


def _pos_to_offset(text: str, line: int, column: int) -> int:
    """Convert line/column (1-indexed) to character offset."""
    lines = text.split('\n')
    offset = sum(len(lines[i]) + 1 for i in range(line - 1))  # +1 for newlines
    return offset + column - 1


def _collect_spans(tree: Tree, source: str) -> list[HighlightSpan]:
    """Walk parse tree and collect highlight spans."""
    spans = []

    def visit(node):
        if isinstance(node, Token):
            css_class = TOKEN_CLASS_MAP.get(node.type)
            if css_class:
                start = _pos_to_offset(source, node.line, node.column)
                end = _pos_to_offset(source, node.end_line, node.end_column)
                spans.append(HighlightSpan(start, end, css_class, str(node)))
        elif isinstance(node, Tree):
            # check if this tree node type should be highlighted as a unit
            css_class = TOKEN_CLASS_MAP.get(node.data)
            if css_class and hasattr(node, 'meta') and node.meta.start_pos is not None:
                # for tree nodes, use meta positions if available
                start = node.meta.start_pos
                end = node.meta.end_pos
                text = source[start:end]
                spans.append(HighlightSpan(start, end, css_class, text))
            else:
                # recurse into children
                for child in node.children:
                    visit(child)

    visit(tree)
    return spans


def _merge_overlapping_spans(spans: list[HighlightSpan]) -> list[HighlightSpan]:
    """Remove spans that are fully contained within other spans.

    When a tree node (like single_completion) covers the same range as its
    tokens, prefer the tree node's class.
    """
    if not spans:
        return []

    # sort by start position, then by length (longer first)
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))

    result = []
    for span in spans:
        # check if this span is contained within any existing span
        contained = False
        for existing in result:
            if span.start >= existing.start and span.end <= existing.end:
                contained = True
                break
        if not contained:
            result.append(span)

    return result


def highlight_struckdown(text: str) -> str:
    """Apply syntax highlighting to struckdown text.

    Uses the Lark parser to identify syntax elements, ensuring highlighting
    matches the actual grammar exactly.

    Returns HTML with spans for each syntax element.
    """
    try:
        tree = _highlight_parser.parse(text)
    except Exception:
        # if parsing fails, return escaped text with no highlighting
        return escape(text)

    spans = _collect_spans(tree, text)
    spans = _merge_overlapping_spans(spans)

    # sort spans by start position
    spans.sort(key=lambda s: s.start)

    # build result HTML
    result = []
    last_end = 0

    for span in spans:
        # add text before this span (escaped)
        if span.start > last_end:
            result.append(escape(text[last_end:span.start]))

        # add highlighted span
        escaped_text = escape(span.text)
        result.append(f'<span class="sd-{span.css_class}">{escaped_text}</span>')
        last_end = span.end

    # add remaining text
    if last_end < len(text):
        result.append(escape(text[last_end:]))

    return ''.join(result)


def highlight_struckdown_with_system_blocks(text: str) -> str:
    """Apply syntax highlighting with special handling for system blocks.

    System blocks get whole-line background styling with:
    - Darker background on tag lines (<system>, </system>)
    - Lighter background on content lines
    """
    try:
        tree = _highlight_parser.parse(text)
    except Exception:
        return escape(text)

    spans = _collect_spans(tree, text)

    # separate system/header-related spans from others
    system_spans = []
    header_spans = []
    other_spans = []

    for span in spans:
        if span.css_class in ('system', 'system-tag', 'system-content'):
            system_spans.append(span)
        elif span.css_class in ('header', 'header-tag', 'header-content'):
            header_spans.append(span)
        else:
            other_spans.append(span)

    # if no system or header blocks, use simple highlighting
    if not system_spans and not header_spans:
        return highlight_struckdown(text)

    # find block boundaries from spans (full blocks)
    system_blocks = [(s, 'system') for s in system_spans if s.css_class == 'system']
    header_blocks = [(s, 'header') for s in header_spans if s.css_class == 'header']
    all_blocks = system_blocks + header_blocks

    if not all_blocks:
        return highlight_struckdown(text)

    # process text with block awareness
    result = []
    last_end = 0

    for block, block_type in sorted(all_blocks, key=lambda x: x[0].start):
        # add text before this block
        if block.start > last_end:
            before_text = text[last_end:block.start]
            result.append(_highlight_segment(before_text, other_spans, last_end))

        # parse the block structure
        block_text = block.text

        # find opening tag
        open_end = block_text.find('>') + 1
        open_tag = block_text[:open_end]

        # find closing tag
        close_start = block_text.rfind('<')
        close_tag = block_text[close_start:]
        content = block_text[open_end:close_start]

        # render block with special styling
        result.append(
            f'<span class="sd-{block_type}-tag-line">'
            f'<span class="sd-{block_type}-tag">{escape(open_tag)}</span>'
            f'</span>'
        )

        if content.strip():
            # highlight inner content using regex fallback since content
            # is captured as a single token without parsing {{vars}} etc
            inner_highlighted = _highlight_unparsed_content(content)
            result.append(f'<span class="sd-{block_type}-content">{inner_highlighted}</span>')
        elif content:
            result.append(f'<span class="sd-{block_type}-content">{escape(content)}</span>')

        result.append(
            f'<span class="sd-{block_type}-tag-line">'
            f'<span class="sd-{block_type}-tag">{escape(close_tag)}</span>'
            f'</span>'
        )

        last_end = block.end

    # add remaining text
    if last_end < len(text):
        result.append(_highlight_segment(text[last_end:], other_spans, last_end))

    return ''.join(result)


def _highlight_unparsed_content(text: str) -> str:
    """Highlight template variables, slots, actions, headings and jinja tags in unparsed content.

    Used for content inside system blocks where the grammar captures
    everything as a single SYSTEM_CONTENT token without parsing inner elements.
    """
    # patterns for elements that should be highlighted (order matters - more specific first)
    patterns = [
        ('break', r'\[\[@break[^\]]*\]\]'),       # break action - red
        ('action', r'\[\[@[^\]]*\]\]'),           # other actions - purple
        ('placeholder', r'\[\[[^\]]*\]\]'),       # slots - green
        ('template-var', r'\{\{[^}]+\}\}'),       # template vars - pink
        ('jinja-tag', r'\{%[^%]+%\}'),
        ('comment', r'<!--[\s\S]*?-->'),
        ('heading', r'^(#{1,6})\s+(.+)$'),        # markdown headings
    ]

    # combine into single pattern with named groups
    combined = '|'.join(f'(?P<{name.replace("-", "_")}>{pattern})' for name, pattern in patterns)

    result = []
    last_end = 0

    for match in re.finditer(combined, text, re.MULTILINE):
        # add text before match
        if match.start() > last_end:
            result.append(escape(text[last_end:match.start()]))

        # find which group matched and apply class
        for name, pattern in patterns:
            group_name = name.replace('-', '_')
            matched_text = match.group(group_name)
            if matched_text is not None:
                if name == 'heading':
                    # special handling for headings - split marker and text
                    heading_match = re.match(r'^(#{1,6})\s+(.+)$', matched_text)
                    if heading_match:
                        marker = heading_match.group(1)
                        heading_text = heading_match.group(2)
                        result.append(
                            f'<span class="sd-heading">'
                            f'<span class="sd-heading-marker">{escape(marker)}</span> '
                            f'<span class="sd-heading-text">{escape(heading_text)}</span>'
                            f'</span>'
                        )
                    else:
                        result.append(f'<span class="sd-heading">{escape(matched_text)}</span>')
                else:
                    result.append(f'<span class="sd-{name}">{escape(matched_text)}</span>')
                break

        last_end = match.end()

    # add remaining text
    if last_end < len(text):
        result.append(escape(text[last_end:]))

    return ''.join(result)


def _highlight_segment(text: str, all_spans: list[HighlightSpan], offset: int) -> str:
    """Highlight a segment of text using spans adjusted for offset."""
    # filter spans that fall within this segment
    segment_end = offset + len(text)
    relevant_spans = [
        HighlightSpan(s.start - offset, s.end - offset, s.css_class, s.text)
        for s in all_spans
        if s.start >= offset and s.end <= segment_end
    ]

    relevant_spans = _merge_overlapping_spans(relevant_spans)
    relevant_spans.sort(key=lambda s: s.start)

    result = []
    last_end = 0

    for span in relevant_spans:
        if span.start > last_end:
            result.append(escape(text[last_end:span.start]))

        escaped_text = escape(span.text)
        result.append(f'<span class="sd-{span.css_class}">{escaped_text}</span>')
        last_end = span.end

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
        /* placeholder [[...]] - green */
        .sd-placeholder {{
            background: #D4EDDA;
            color: #155724;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid #c3e6cb;
        }}
        /* action slots [[@...]] - purple */
        .sd-action {{
            background: #EDE7F6;
            color: #6A1B9A;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid #d1c4e9;
        }}
        /* break action [[@break]] - red */
        .sd-break {{
            background: #FCDCDC;
            color: #F23030;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid #f5c6cb;
        }}
        /* template variables {{...}} - pink */
        .sd-template-var {{
            background: #FBDCE8;
            color: #9B2C5A;
            font-weight: bold;
            border-radius: 4px;
            padding: 2px 6px;
            border: 1px solid #f8c9dc;
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
        /* header tag line - faded post-it yellow */
        .sd-header-tag-line {{
            background: #fef9e7;
            display: block;
            padding: 4px 12px;
            margin: 0 -1.5em;
            padding-left: 1.5em;
            padding-right: 1.5em;
            border-left: 4px solid #f5deb3;
        }}
        /* header tag text - muted brown */
        .sd-header-tag {{
            font-weight: bold;
            color: #8b7355;
        }}
        /* header content - very faint yellow */
        .sd-header-content {{
            background: #fffdf5;
            display: block;
            padding: 8px 12px;
            margin: 0 -1.5em;
            padding-left: 1.5em;
            padding-right: 1.5em;
            border-left: 4px solid #f5e6c8;
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
        /* comments - muted */
        .sd-comment {{
            color: #888888;
            font-style: italic;
            opacity: 0.8;
        }}
        /* markdown headings */
        .sd-heading {{
            font-weight: bold;
        }}
        .sd-heading-marker {{
            color: #BF360C;
            font-weight: bold;
        }}
        .sd-heading-text {{
            color: #1565C0;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>{escape(filename)}</h1>
    <pre>{highlighted}</pre>
</body>
</html>'''
