"""Reading input files: documents go through extraction, text through fallback encodings."""

import zipfile
from pathlib import Path

import pytest

from struckdown.sd_cli import _read_document_text, _read_input_file

DOCX_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>Hello from a docx</w:t></w:r></w:p></w:body></w:document>
"""

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def _make_docx(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", DOCX_XML)
    return path


def test_docx_is_extracted_not_decoded_as_utf8(tmp_path):
    """A .docx is a zip -- reading it as utf-8 used to raise UnicodeDecodeError."""
    pytest.importorskip("kreuzberg")
    path = _make_docx(tmp_path / "doc.docx")
    items = _read_input_file(path)
    assert len(items) == 1
    assert "Hello from a docx" in items[0]["content"]
    assert items[0]["basename"] == "doc"


def test_text_file_with_non_utf8_bytes_falls_back(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes("caf\xe9 na\xefve".encode("cp1252"))
    assert "caf" in _read_document_text(path)


def test_unknown_extension_reads_as_plain_text(tmp_path):
    path = tmp_path / "prompt.sd"
    path.write_text("[[speak:answer]]")
    assert _read_document_text(path) == "[[speak:answer]]"
