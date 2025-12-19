"""
Disk-based cache for evidence files in the playground.

Stores uploaded evidence documents per session for BM25 search.
Uses FIFO eviction when cache size limit is exceeded.

Configuration via environment variables:
- STRUCKDOWN_EVIDENCE_CACHE_DIR: Cache directory (default: ~/.struckdown/evidence)
- STRUCKDOWN_EVIDENCE_CACHE_SIZE: Max cache size in MB (default: 500)
- STRUCKDOWN_EVIDENCE_MAX_AGE: Max file age in seconds (default: 86400 = 1 day)
"""

import json
import logging
import os
import time
from pathlib import Path

import validators

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/evidence").expanduser()
EVIDENCE_CACHE_DIR = Path(
    os.environ.get("STRUCKDOWN_EVIDENCE_CACHE_DIR", str(_default_cache_dir))
)
EVIDENCE_CACHE_SIZE_MB = int(
    os.environ.get("STRUCKDOWN_EVIDENCE_CACHE_SIZE", "500")
)  # 500MB
EVIDENCE_MAX_AGE_SECONDS = int(
    os.environ.get("STRUCKDOWN_EVIDENCE_MAX_AGE", "86400")
)  # 1 day


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    EVIDENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str) -> Path:
    """Get the directory for a session's evidence files."""
    return EVIDENCE_CACHE_DIR / session_id


def validate_session_id(session_id: str) -> bool:
    """Validate session_id format (alphanumeric + underscores, reasonable length)."""
    if not session_id or len(session_id) > 100:
        return False
    return all(c.isalnum() or c in "_-" for c in session_id)


def validate_file_id(file_id: str) -> bool:
    """Validate file_id is a proper UUID."""
    return validators.uuid(file_id) is True


def store_evidence(
    session_id: str, file_id: str, filename: str, chunks: list[dict]
) -> None:
    """Store evidence file with chunks to disk cache.

    Args:
        session_id: Session identifier
        file_id: UUID string identifying the file
        filename: Original filename
        chunks: List of chunk dicts with text, index, start_char, end_char

    Raises:
        ValueError: If session_id or file_id is invalid
    """
    if not validate_session_id(session_id):
        raise ValueError(f"Invalid session_id: {session_id}")
    if not validate_file_id(file_id):
        raise ValueError(f"Invalid file_id: {file_id}")

    _ensure_cache_dir()
    session_path = _session_dir(session_id)
    session_path.mkdir(parents=True, exist_ok=True)

    cache_data = {
        "file_id": file_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "uploaded_at": time.time(),
    }
    path = session_path / f"{file_id}.json"

    # Atomic write
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache_data))
    tmp_path.rename(path)

    logger.debug(f"Stored evidence {file_id} for session {session_id}")


def get_evidence_for_session(session_id: str) -> list[dict]:
    """Get all evidence chunks for a session (for BM25 search).

    Args:
        session_id: Session identifier

    Returns:
        List of chunk dicts with text, filename, index
    """
    if not validate_session_id(session_id):
        return []

    session_path = _session_dir(session_id)
    if not session_path.exists():
        return []

    now = time.time()
    all_chunks = []

    for path in session_path.glob("*.json"):
        try:
            data = json.loads(path.read_text())

            # Skip expired
            if now - data.get("uploaded_at", 0) > EVIDENCE_MAX_AGE_SECONDS:
                continue

            filename = data.get("filename", "unknown")
            for chunk in data.get("chunks", []):
                all_chunks.append(
                    {
                        "text": chunk["text"],
                        "filename": filename,
                        "index": chunk["index"],
                        "file_id": data["file_id"],
                    }
                )
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return all_chunks


def list_evidence(session_id: str) -> list[dict]:
    """List metadata for all evidence files in a session.

    Args:
        session_id: Session identifier

    Returns:
        List of dicts with file_id, filename, chunk_count, uploaded_at
    """
    if not validate_session_id(session_id):
        return []

    session_path = _session_dir(session_id)
    if not session_path.exists():
        return []

    now = time.time()
    files = []

    for path in session_path.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            uploaded_at = data.get("uploaded_at", 0)

            # Skip expired
            if now - uploaded_at > EVIDENCE_MAX_AGE_SECONDS:
                continue

            files.append(
                {
                    "file_id": data.get("file_id"),
                    "filename": data.get("filename"),
                    "chunk_count": data.get("chunk_count", 0),
                    "uploaded_at": uploaded_at,
                }
            )
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    # Sort by upload time, newest first
    files.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return files


def delete_evidence(session_id: str, file_id: str) -> bool:
    """Delete an evidence file from a session.

    Args:
        session_id: Session identifier
        file_id: UUID string identifying the file

    Returns:
        True if file was deleted, False if not found
    """
    if not validate_session_id(session_id) or not validate_file_id(file_id):
        return False

    path = _session_dir(session_id) / f"{file_id}.json"
    if path.exists():
        path.unlink()
        logger.debug(f"Deleted evidence {file_id} from session {session_id}")
        return True
    return False


def cleanup_cache() -> int:
    """Remove expired files and empty session directories.

    Returns:
        Number of files removed
    """
    if not EVIDENCE_CACHE_DIR.exists():
        return 0

    now = time.time()
    removed = 0

    # Process each session directory
    for session_path in EVIDENCE_CACHE_DIR.iterdir():
        if not session_path.is_dir():
            continue

        for path in session_path.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                uploaded_at = data.get("uploaded_at", 0)

                if now - uploaded_at > EVIDENCE_MAX_AGE_SECONDS:
                    path.unlink()
                    removed += 1
                    logger.debug(f"Removed expired evidence: {path.name}")
            except (json.JSONDecodeError, OSError):
                path.unlink(missing_ok=True)
                removed += 1

        # Remove empty session directories
        if session_path.exists() and not any(session_path.iterdir()):
            session_path.rmdir()
            logger.debug(f"Removed empty session dir: {session_path.name}")

    if removed > 0:
        logger.info(f"Evidence cache cleanup: removed {removed} files")

    return removed
