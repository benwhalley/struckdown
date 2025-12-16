#!/usr/bin/env python3
"""Standalone script to highlight SD syntax without heavy dependencies."""

import sys
from html import escape
from pathlib import Path

# Inline the minimal highlighting logic to avoid importing struckdown package
# This only needs lark, which is a lightweight dependency


def get_input():
    """Get input from file arg or stdin."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text(encoding="utf-8")
    return sys.stdin.read()


try:
    from lark import Lark, Token, Tree
except ImportError:
    # Fallback: just escape and output
    print(escape(get_input()), end="")
    sys.exit(0)

# Load grammar from the same directory structure
SCRIPT_DIR = Path(__file__).parent.parent.parent
GRAMMAR_PATH = SCRIPT_DIR / "grammar.lark"

if not GRAMMAR_PATH.exists():
    # Fallback
    print(escape(get_input()), end="")
    sys.exit(0)

_grammar_text = GRAMMAR_PATH.read_text(encoding="utf-8")
_highlight_parser = Lark(_grammar_text, parser="lalr", propagate_positions=True)

TOKEN_CLASS_MAP = {
    "block_comment": "comment",
    "BLOCK_COMMENT_START": "comment",
    "BLOCK_COMMENT_CONTENT": "comment",
    "BLOCK_COMMENT_END": "comment",
    "checkpoint_closed": "checkpoint",
    "checkpoint_auto": "checkpoint",
    "CHECKPOINT_CLOSED": "checkpoint",
    "CHECKPOINT_OPEN": "checkpoint",
    "obliviate_closed": "obliviate",
    "obliviate_auto": "obliviate",
    "OBLIVIATE_CLOSED": "obliviate",
    "OBLIVIATE_OPEN": "obliviate",
    "system_tag": "system",
    "SYSTEM_OPEN": "system-tag",
    "SYSTEM_CLOSE": "system-tag",
    "SYSTEM_CONTENT": "system-content",
    "include_tag": "include",
    "INCLUDE_SRC": "include",
    "single_completion": "placeholder",
    "placeholder": "template-var",
    "templatetag": "jinja-tag",
}


class HighlightSpan:
    __slots__ = ("start", "end", "css_class", "text")

    def __init__(self, start, end, css_class, text):
        self.start = start
        self.end = end
        self.css_class = css_class
        self.text = text


def _pos_to_offset(text, line, column):
    lines = text.split("\n")
    offset = sum(len(lines[i]) + 1 for i in range(line - 1))
    return offset + column - 1


def _collect_spans(tree, source):
    spans = []

    def visit(node):
        if isinstance(node, Token):
            css_class = TOKEN_CLASS_MAP.get(node.type)
            if css_class:
                start = _pos_to_offset(source, node.line, node.column)
                end = _pos_to_offset(source, node.end_line, node.end_column)
                spans.append(HighlightSpan(start, end, css_class, str(node)))
        elif isinstance(node, Tree):
            css_class = TOKEN_CLASS_MAP.get(node.data)
            if css_class and hasattr(node, "meta") and node.meta.start_pos is not None:
                start = node.meta.start_pos
                end = node.meta.end_pos
                text = source[start:end]
                spans.append(HighlightSpan(start, end, css_class, text))
            else:
                for child in node.children:
                    visit(child)

    visit(tree)
    return spans


def _merge_overlapping_spans(spans):
    if not spans:
        return []
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    result = []
    for span in spans:
        contained = False
        for existing in result:
            if span.start >= existing.start and span.end <= existing.end:
                contained = True
                break
        if not contained:
            result.append(span)
    return result


def highlight_struckdown_with_system_blocks(text):
    try:
        tree = _highlight_parser.parse(text)
    except Exception:
        return escape(text)

    spans = _collect_spans(tree, text)
    system_spans = []
    other_spans = []

    for span in spans:
        if span.css_class in ("system", "system-tag", "system-content"):
            system_spans.append(span)
        else:
            other_spans.append(span)

    if not system_spans:
        return _highlight_simple(text, spans)

    system_blocks = [s for s in system_spans if s.css_class == "system"]
    if not system_blocks:
        return _highlight_simple(text, spans)

    result = []
    last_end = 0

    for block in sorted(system_blocks, key=lambda s: s.start):
        if block.start > last_end:
            before_text = text[last_end : block.start]
            result.append(_highlight_segment(before_text, other_spans, last_end))

        block_text = block.text
        open_end = block_text.find(">") + 1
        open_tag = block_text[:open_end]
        close_start = block_text.rfind("<")
        close_tag = block_text[close_start:]
        content = block_text[open_end:close_start]

        result.append(
            f'<span class="sd-system-tag-line"><span class="sd-system-tag">{escape(open_tag)}</span></span>'
        )

        if content.strip():
            inner_highlighted = _highlight_segment(
                content, other_spans, block.start + open_end
            )
            result.append(f'<span class="sd-system-content">{inner_highlighted}</span>')
        elif content:
            result.append(f'<span class="sd-system-content">{escape(content)}</span>')

        result.append(
            f'<span class="sd-system-tag-line"><span class="sd-system-tag">{escape(close_tag)}</span></span>'
        )

        last_end = block.end

    if last_end < len(text):
        result.append(_highlight_segment(text[last_end:], other_spans, last_end))

    return "".join(result)


def _highlight_simple(text, spans):
    spans = _merge_overlapping_spans(spans)
    spans.sort(key=lambda s: s.start)
    result = []
    last_end = 0
    for span in spans:
        if span.start > last_end:
            result.append(escape(text[last_end : span.start]))
        escaped_text = escape(span.text)
        result.append(f'<span class="sd-{span.css_class}">{escaped_text}</span>')
        last_end = span.end
    if last_end < len(text):
        result.append(escape(text[last_end:]))
    return "".join(result)


def _highlight_segment(text, all_spans, offset):
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
            result.append(escape(text[last_end : span.start]))
        escaped_text = escape(span.text)
        result.append(f'<span class="sd-{span.css_class}">{escaped_text}</span>')
        last_end = span.end
    if last_end < len(text):
        result.append(escape(text[last_end:]))
    return "".join(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        content = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()
    print(highlight_struckdown_with_system_blocks(content), end="")
