"""Document text extraction.

Public API:
- :func:`extract_text` -- returns markdown string for documents (PDF, DOCX, RTF,
  txt, md, html, images via OCR, etc.). Uses kreuzberg by default. If
  ``prefer_pandoc=True`` and pypandoc + the pandoc binary are both available,
  routes ``.docx``/``.rtf``/``.odt``/``.epub`` through pandoc for cleaner GFM
  output (preserves italics, headings, tables) and falls back to kreuzberg
  silently when pandoc is unavailable.
- :func:`extract_rows` -- returns ``list[dict]`` for CSV/XLSX/XLS spreadsheets,
  one dict per row keyed by column name.

Install the ``extract`` extra (kreuzberg only, ~64 MB, no system deps)::

    pip install struckdown[extract]

For pandoc-quality DOCX extraction, separately ``pip install pypandoc`` and
install the ``pandoc`` system binary.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".vtt", ".srt", ".log", ".rst"}
PANDOC_EXTENSIONS = {".docx", ".rtf", ".odt", ".epub"}
SPREADSHEET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class ExtractionError(RuntimeError):
    """Raised when document extraction fails or required deps are missing."""


def _normalise_whitespace(text: str) -> str:
    """Strip nulls, normalise line endings, collapse runs of blank lines."""
    text = text.replace("\u0000", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text_file(path: Path) -> str:
    """Read plain text with encoding fallback chain."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _pandoc_to_gfm(path: Path) -> str | None:
    """Convert via pandoc to GFM. Returns None if pandoc/pypandoc unavailable."""
    try:
        import pypandoc
    except ImportError:
        return None
    try:
        return pypandoc.convert_file(
            str(path),
            to="gfm",
            format=None,
            extra_args=["--wrap=none", "--strip-comments"],
        )
    except (OSError, RuntimeError) as e:
        logger.warning("pandoc conversion failed for %s: %s; falling back", path, e)
        return None


def _kreuzberg_extract(path: Path) -> str:
    try:
        from kreuzberg import extract_file_sync
    except ImportError as e:
        raise ExtractionError(
            "kreuzberg is required for document extraction. "
            "Install with: pip install struckdown[extract]"
        ) from e
    result = extract_file_sync(str(path))
    return getattr(result, "content", str(result))


def extract_text(path: str | Path, *, prefer_pandoc: bool = True) -> str:
    """Extract text from a document, returning a markdown string.

    Args:
        path: Path to a PDF, DOCX, RTF, ODT, EPUB, image, HTML, or plain text file.
        prefer_pandoc: When True (default) and the file is ``.docx``/``.rtf``/etc.,
            try pandoc first for cleaner GFM output (italics, headings, tables
            preserved). Falls back to kreuzberg if pandoc is unavailable.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in SPREADSHEET_EXTENSIONS:
        raise ExtractionError(
            f"{suffix} is a spreadsheet -- use extract_rows() instead"
        )

    if suffix in TEXT_EXTENSIONS:
        return _normalise_whitespace(_read_text_file(p))

    if prefer_pandoc and suffix in PANDOC_EXTENSIONS:
        text = _pandoc_to_gfm(p)
        if text is not None:
            return _normalise_whitespace(text)

    return _normalise_whitespace(_kreuzberg_extract(p))


def extract_rows(path: str | Path) -> list[dict[str, Any]]:
    """Extract rows from CSV/XLSX/XLS as a list of dicts (one per row).

    NaN values are converted to None. Column names from the header row become
    dict keys.
    """
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix == ".xlsx":
        df = pd.read_excel(p, engine="openpyxl")
    elif suffix == ".xls":
        df = pd.read_excel(p)
    else:
        raise ExtractionError(f"Unsupported spreadsheet format: {suffix}")

    return df.where(pd.notna(df), None).to_dict("records")


def is_spreadsheet(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SPREADSHEET_EXTENSIONS


def is_supported(path: str | Path) -> bool:
    """True if the extension is one ``extract_text`` or ``extract_rows`` handles
    natively. Other extensions still attempt kreuzberg, which covers more formats."""
    suffix = Path(path).suffix.lower()
    return (
        suffix in TEXT_EXTENSIONS
        or suffix in PANDOC_EXTENSIONS
        or suffix in SPREADSHEET_EXTENSIONS
        or suffix == ".pdf"
    )
