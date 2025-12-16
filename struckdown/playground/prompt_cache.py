"""
Disk-based storage for prompts in the playground.

Stores prompt text as plain .txt files on disk.
Prompts persist indefinitely (no expiry, no size limit).

Configuration via environment variables:
- STRUCKDOWN_PROMPT_CACHE_DIR: Storage directory (default: ~/.struckdown/prompts)
"""

import logging
import os
from pathlib import Path

import validators

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/prompts").expanduser()
PROMPT_CACHE_DIR = Path(
    os.environ.get("STRUCKDOWN_PROMPT_CACHE_DIR", str(_default_cache_dir))
)


def _ensure_cache_dir() -> None:
    """Ensure the storage directory exists."""
    PROMPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def validate_prompt_id(prompt_id: str) -> bool:
    """Validate prompt_id is a proper UUID using validators package."""
    return validators.uuid(prompt_id) is True


def store_prompt(prompt_id: str, text: str) -> None:
    """Store prompt text to disk.

    Args:
        prompt_id: UUID string identifying the prompt
        text: Raw prompt text (struckdown syntax)

    Raises:
        ValueError: If prompt_id is not a valid UUID
    """
    if not validate_prompt_id(prompt_id):
        raise ValueError(f"Invalid prompt_id: {prompt_id}")

    _ensure_cache_dir()

    path = PROMPT_CACHE_DIR / f"{prompt_id}.txt"

    # Atomic write: write to temp file then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.rename(path)

    logger.debug(f"Stored prompt {prompt_id} to {path}")


def get_prompt(prompt_id: str) -> str:
    """Retrieve prompt text from disk.

    Args:
        prompt_id: UUID string identifying the prompt

    Returns:
        Raw prompt text

    Raises:
        ValueError: If prompt_id is not a valid UUID
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
        prompt_id: UUID string identifying the prompt

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
