"""
Caching utilities for struckdown using joblib.Memory.

Cache can be controlled via STRUCKDOWN_CACHE environment variable:
- Default: ~/.struckdown/cache
- Disable: Set to "0", "false", or empty string
- Custom location: Set to any valid directory path
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from decouple import config as env_config
from joblib import Memory

logger = logging.getLogger(__name__)


def get_cache_dir() -> Optional[Path]:
    """
    Get the cache directory from environment variable.

    Returns None if caching is disabled.
    """
    cache_setting = env_config("STRUCKDOWN_CACHE", default="~/.struckdown/cache")

    # Check if caching is disabled
    if cache_setting.lower() in ("0", "false", ""):
        return None

    # Expand user path and return
    return Path(cache_setting).expanduser()


def hash_return_type(return_type: Any) -> str:
    """
    Create a stable hash for a return_type (Pydantic model or callable).

    For Pydantic models, we use the schema.
    For callables, we use the function name and module.
    """
    if hasattr(return_type, "model_json_schema"):
        # Pydantic model - use its schema
        schema_str = str(return_type.model_json_schema())
        return hashlib.md5(schema_str.encode()).hexdigest()
    elif callable(return_type):
        # Function - use its qualified name
        qual_name = f"{return_type.__module__}.{return_type.__qualname__}"
        return hashlib.md5(qual_name.encode()).hexdigest()
    else:
        # Fallback to string representation
        return hashlib.md5(str(return_type).encode()).hexdigest()


# Initialize the Memory object
cache_dir = get_cache_dir()
memory = None

if cache_dir is not None:
    try:
        # Try to create the cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        memory = Memory(location=str(cache_dir), verbose=0)
    except (PermissionError, OSError) as e:
        # If we can't access the configured cache dir, try local fallback
        try:
            fallback_dir = Path(".struckdown/cache")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            memory = Memory(location=str(fallback_dir), verbose=0)
        except (PermissionError, OSError):
            # If both fail, disable caching
            memory = Memory(location=None, verbose=0)
else:
    # Caching disabled - use a no-op Memory object
    memory = Memory(location=None, verbose=0)


def clear_cache():
    """
    Clear all cached results by removing the cache directory.

    This is useful if cache files become corrupted or you want to force
    fresh LLM calls for all prompts.
    """
    cache_dir = get_cache_dir()
    if cache_dir is None:
        logger.info("Caching is disabled, nothing to clear")
        return

    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache directory: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clear cache directory {cache_dir}: {e}")
    else:
        logger.info(f"Cache directory does not exist: {cache_dir}")
