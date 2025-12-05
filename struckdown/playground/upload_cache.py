"""
Disk-based cache for uploaded files in the playground.

Stores uploaded file data as JSON files on disk instead of in memory.
Uses FIFO eviction when cache size limit is exceeded.

Configuration via environment variables:
- STRUCKDOWN_UPLOAD_CACHE_DIR: Cache directory (default: ~/.struckdown/uploads)
- STRUCKDOWN_UPLOAD_CACHE_SIZE: Max cache size in MB (default: 1024 = 1GB)
- STRUCKDOWN_UPLOAD_MAX_AGE: Max file age in seconds (default: 86400 = 1 day)
"""

import json
import logging
import os
import time
from pathlib import Path

import validators

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/uploads").expanduser()
UPLOAD_CACHE_DIR = Path(os.environ.get("STRUCKDOWN_UPLOAD_CACHE_DIR", str(_default_cache_dir)))
UPLOAD_CACHE_SIZE_MB = int(os.environ.get("STRUCKDOWN_UPLOAD_CACHE_SIZE", "1024"))  # 1GB
UPLOAD_MAX_AGE_SECONDS = int(os.environ.get("STRUCKDOWN_UPLOAD_MAX_AGE", "86400"))  # 1 day


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def validate_file_id(file_id: str) -> bool:
    """Validate file_id is a proper UUID using validators package."""
    return validators.uuid(file_id) is True


def store_upload(file_id: str, data: dict) -> None:
    """Store upload data to disk cache.

    Args:
        file_id: UUID string identifying the upload
        data: Dictionary containing upload data (filename, content/rows, etc.)

    Raises:
        ValueError: If file_id is not a valid UUID
    """
    if not validate_file_id(file_id):
        raise ValueError(f"Invalid file_id: {file_id}")

    _ensure_cache_dir()

    cache_data = {**data, "uploaded_at": time.time()}
    path = UPLOAD_CACHE_DIR / f"{file_id}.json"

    # Atomic write: write to temp file then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(cache_data))
    tmp_path.rename(path)

    logger.debug(f"Stored upload {file_id} to {path}")


def get_upload(file_id: str) -> dict:
    """Retrieve upload from disk cache.

    Args:
        file_id: UUID string identifying the upload

    Returns:
        Dictionary containing the upload data

    Raises:
        ValueError: If file_id is not a valid UUID
        FileNotFoundError: If upload not found or expired
    """
    if not validate_file_id(file_id):
        raise ValueError(f"Invalid file_id: {file_id}")

    path = UPLOAD_CACHE_DIR / f"{file_id}.json"
    if not path.exists():
        raise FileNotFoundError("Upload not found or expired")

    data = json.loads(path.read_text())

    # Check expiry
    if time.time() - data.get("uploaded_at", 0) > UPLOAD_MAX_AGE_SECONDS:
        path.unlink(missing_ok=True)
        logger.debug(f"Upload {file_id} expired, removed")
        raise FileNotFoundError("Upload expired")

    return data


def delete_upload(file_id: str) -> bool:
    """Delete an upload from the cache.

    Args:
        file_id: UUID string identifying the upload

    Returns:
        True if file was deleted, False if it didn't exist
    """
    if not validate_file_id(file_id):
        return False

    path = UPLOAD_CACHE_DIR / f"{file_id}.json"
    if path.exists():
        path.unlink()
        logger.debug(f"Deleted upload {file_id}")
        return True
    return False


def cleanup_cache() -> int:
    """Remove expired files and enforce size limit using FIFO.

    Returns:
        Number of files removed
    """
    if not UPLOAD_CACHE_DIR.exists():
        return 0

    now = time.time()
    files = []
    removed = 0

    # Collect all cache files with their timestamps
    for path in UPLOAD_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            uploaded_at = data.get("uploaded_at", 0)

            # Remove expired
            if now - uploaded_at > UPLOAD_MAX_AGE_SECONDS:
                path.unlink()
                removed += 1
                logger.debug(f"Removed expired upload: {path.stem}")
                continue

            files.append((path, uploaded_at, path.stat().st_size))
        except (json.JSONDecodeError, OSError):
            # Remove corrupted files
            path.unlink(missing_ok=True)
            removed += 1

    # Enforce size limit with FIFO (oldest first)
    total_size = sum(f[2] for f in files)
    size_limit = UPLOAD_CACHE_SIZE_MB * 1024 * 1024

    if total_size > size_limit:
        # Sort by upload time (oldest first)
        files.sort(key=lambda x: x[1])
        for path, _, size in files:
            if total_size <= size_limit:
                break
            path.unlink(missing_ok=True)
            total_size -= size
            removed += 1
            logger.debug(f"Removed upload for size limit: {path.stem}")

    if removed > 0:
        logger.info(f"Cache cleanup: removed {removed} files")

    return removed


def get_cache_stats() -> dict:
    """Get statistics about the upload cache.

    Returns:
        Dictionary with file_count, total_size_mb, oldest_file_age_seconds
    """
    if not UPLOAD_CACHE_DIR.exists():
        return {"file_count": 0, "total_size_mb": 0, "oldest_file_age_seconds": 0}

    now = time.time()
    files = list(UPLOAD_CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files if f.exists())

    oldest_age = 0
    for path in files:
        try:
            data = json.loads(path.read_text())
            age = now - data.get("uploaded_at", now)
            oldest_age = max(oldest_age, age)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "oldest_file_age_seconds": round(oldest_age),
    }
