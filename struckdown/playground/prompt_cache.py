"""
Disk-based storage for prompts in the playground.

Stores prompt text as plain .txt files on disk, using content-based
SHA256 hashing for filenames. Same content always produces same hash/URL.

Files are stored with restricted permissions (0600).

LRU eviction: access time is updated on read, oldest-accessed files
are deleted first when cache exceeds size limit.

Configuration via environment variables:
- STRUCKDOWN_PROMPT_CACHE_DIR: Storage directory (default: ~/.struckdown/prompts)
- STRUCKDOWN_PROMPT_MAX_SIZE: Max single prompt size in bytes (default: 1MB)
- STRUCKDOWN_PROMPT_CACHE_MAX_SIZE: Max total cache size in bytes (default: 100MB, 0=unlimited)
"""

import hashlib
import logging
import os
import re
import secrets
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/prompts").expanduser()
PROMPT_CACHE_DIR = Path(
    os.environ.get("STRUCKDOWN_PROMPT_CACHE_DIR", str(_default_cache_dir))
)

# Use first 24 chars of SHA256 (96 bits) -- collision-resistant for this use case
HASH_LENGTH = 24

# Max single prompt size (default 1MB)
MAX_PROMPT_SIZE = int(os.environ.get("STRUCKDOWN_PROMPT_MAX_SIZE", 1 * 1024 * 1024))

# Max total cache size (default 100MB, 0 = unlimited)
MAX_CACHE_SIZE = int(os.environ.get("STRUCKDOWN_PROMPT_CACHE_MAX_SIZE", 100 * 1024 * 1024))


def _ensure_cache_dir() -> None:
    """Ensure the storage directory exists with restricted permissions."""
    PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)


def hash_content(text: str) -> str:
    """Generate a content-based hash for the given text.

    Returns first HASH_LENGTH characters of SHA256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def validate_prompt_id(prompt_id: str) -> bool:
    """Validate prompt_id is a valid hex hash of expected length."""
    if not prompt_id or len(prompt_id) != HASH_LENGTH:
        return False
    return bool(re.match(r"^[a-f0-9]+$", prompt_id))


def store_prompt(text: str) -> str:
    """Store prompt text to disk using content-based hash.

    Args:
        text: Raw prompt text (struckdown syntax)

    Returns:
        The content hash (prompt_id) for this prompt

    Raises:
        ValueError: If prompt exceeds MAX_PROMPT_SIZE
    """
    # Check size limit
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_PROMPT_SIZE:
        raise ValueError(f"Prompt too large ({len(text_bytes)} bytes, max {MAX_PROMPT_SIZE})")

    prompt_id = hash_content(text)

    _ensure_cache_dir()

    path = PROMPT_CACHE_DIR / f"{prompt_id}.txt"

    # If file already exists with same hash, touch to update access time (LRU)
    if path.exists():
        path.touch()
        logger.debug(f"Prompt {prompt_id} already exists at {path}")
        return prompt_id

    # Atomic write: use unique temp file to avoid race conditions
    tmp_suffix = secrets.token_hex(8)
    tmp_path = PROMPT_CACHE_DIR / f"{prompt_id}.{tmp_suffix}.tmp"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.chmod(tmp_path, 0o600)
        tmp_path.rename(path)
        logger.debug(f"Stored prompt {prompt_id} to {path}")
    except FileExistsError:
        # Another request already created the file - that's fine
        tmp_path.unlink(missing_ok=True)
        logger.debug(f"Prompt {prompt_id} already exists (race)")
    except Exception:
        # Clean up temp file on any error
        tmp_path.unlink(missing_ok=True)
        raise

    return prompt_id


def get_prompt(prompt_id: str) -> str:
    """Retrieve prompt text from disk.

    Updates access time for LRU eviction.

    Args:
        prompt_id: Content hash identifying the prompt

    Returns:
        Raw prompt text

    Raises:
        ValueError: If prompt_id is not a valid hash
        FileNotFoundError: If prompt not found
    """
    if not validate_prompt_id(prompt_id):
        raise ValueError(f"Invalid prompt_id: {prompt_id}")

    path = PROMPT_CACHE_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise FileNotFoundError("Prompt not found")

    # Touch to update access time (LRU)
    path.touch()

    return path.read_text(encoding="utf-8")


def prompt_exists(prompt_id: str) -> bool:
    """Check if a prompt exists.

    Args:
        prompt_id: Content hash identifying the prompt

    Returns:
        True if prompt exists, False otherwise
    """
    if not validate_prompt_id(prompt_id):
        return False

    path = PROMPT_CACHE_DIR / f"{prompt_id}.txt"
    return path.exists()


def get_cache_stats() -> dict:
    """Get statistics about the prompt storage.

    Returns:
        Dictionary with file_count, total_size_mb, max_size_mb
    """
    if not PROMPT_CACHE_DIR.exists():
        return {"file_count": 0, "total_size_mb": 0, "max_size_mb": MAX_CACHE_SIZE / (1024 * 1024)}

    files = list(PROMPT_CACHE_DIR.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in files if f.exists())

    return {
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "max_size_mb": round(MAX_CACHE_SIZE / (1024 * 1024), 2) if MAX_CACHE_SIZE > 0 else None,
    }


def cleanup_cache() -> dict:
    """Remove least-recently-used files until cache is under size limit.

    Uses file modification time (updated on access via touch) for LRU ordering.

    Returns:
        Dictionary with files_deleted, bytes_freed
    """
    if MAX_CACHE_SIZE <= 0:
        return {"files_deleted": 0, "bytes_freed": 0}

    if not PROMPT_CACHE_DIR.exists():
        return {"files_deleted": 0, "bytes_freed": 0}

    # Get all prompt files with their stats
    files_with_stats = []
    for f in PROMPT_CACHE_DIR.glob("*.txt"):
        try:
            stat = f.stat()
            files_with_stats.append((f, stat.st_size, stat.st_mtime))
        except OSError:
            continue

    # Calculate total size
    total_size = sum(size for _, size, _ in files_with_stats)

    if total_size <= MAX_CACHE_SIZE:
        logger.debug(f"Cache size {total_size} is under limit {MAX_CACHE_SIZE}")
        return {"files_deleted": 0, "bytes_freed": 0}

    # Sort by mtime ascending (oldest first = least recently used)
    files_with_stats.sort(key=lambda x: x[2])

    files_deleted = 0
    bytes_freed = 0

    # Delete oldest files until under limit
    for path, size, mtime in files_with_stats:
        if total_size <= MAX_CACHE_SIZE:
            break

        try:
            path.unlink()
            total_size -= size
            bytes_freed += size
            files_deleted += 1
            logger.info(f"Deleted LRU prompt: {path.name} ({size} bytes)")
        except OSError as e:
            logger.warning(f"Failed to delete {path}: {e}")

    logger.info(f"Cache cleanup: deleted {files_deleted} files, freed {bytes_freed} bytes")
    return {"files_deleted": files_deleted, "bytes_freed": bytes_freed}
