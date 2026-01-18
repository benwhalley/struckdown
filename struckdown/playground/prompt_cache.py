"""
Disk-based storage for prompts in the playground.

Stores prompt text as plain .txt files on disk, using content-based
SHA256 hashing for filenames. Same content always produces same hash/URL.

Files are stored with restricted permissions (0600).

Configuration via environment variables:
- STRUCKDOWN_PROMPT_CACHE_DIR: Storage directory (default: ~/.struckdown/prompts)
"""

import hashlib
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/prompts").expanduser()
PROMPT_CACHE_DIR = Path(
    os.environ.get("STRUCKDOWN_PROMPT_CACHE_DIR", str(_default_cache_dir))
)

# Use first 24 chars of SHA256 (96 bits) -- collision-resistant for this use case
HASH_LENGTH = 24


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
    """
    prompt_id = hash_content(text)

    _ensure_cache_dir()

    path = PROMPT_CACHE_DIR / f"{prompt_id}.txt"

    # If file already exists with same hash, no need to rewrite
    if path.exists():
        logger.debug(f"Prompt {prompt_id} already exists at {path}")
        return prompt_id

    # Atomic write: write to temp file then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    # Set restrictive permissions (owner read/write only)
    os.chmod(tmp_path, 0o600)
    tmp_path.rename(path)

    logger.debug(f"Stored prompt {prompt_id} to {path}")
    return prompt_id


def get_prompt(prompt_id: str) -> str:
    """Retrieve prompt text from disk.

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
        Dictionary with file_count, total_size_mb
    """
    if not PROMPT_CACHE_DIR.exists():
        return {"file_count": 0, "total_size_mb": 0}

    files = list(PROMPT_CACHE_DIR.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in files if f.exists())

    return {
        "file_count": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }
