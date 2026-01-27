"""
Caching utilities for struckdown using joblib.Memory.

Cache can be controlled via environment variables:

STRUCKDOWN_CACHE:
- Default: ~/.struckdown/cache
- Disable: Set to "0", "false", or empty string
- Custom location: Set to any valid directory path

STRUCKDOWN_CACHE_SIZE:
- Default: 10240 (MB) = 10 GB
- Set to desired cache size limit in megabytes
- Cache is reduced to this size when the module is loaded (keeps most recently accessed items)
- NOTE: Cache can grow beyond this limit during runtime; size limit is only enforced at module load
- Set to 0 for unlimited cache (not recommended)
"""

import hashlib
import logging
import os
import shutil
import threading
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


def get_cache_size_limit() -> Optional[int]:
    """
    Get the cache size limit in bytes from environment variable.

    Returns None for unlimited cache, or bytes limit for LRU eviction.
    Default: 10240 MB (10 GB)
    """
    size_mb = env_config("STRUCKDOWN_CACHE_SIZE", default=10240, cast=int)

    # 0 means unlimited
    if size_mb == 0:
        return None

    # Convert MB to bytes
    return size_mb * 1024 * 1024


def hash_return_type(return_type: Any) -> str:
    """
    Create a stable hash for a return_type (Pydantic model or callable).

    For Pydantic models, we use the schema plus llm_config and YAML definition hash.
    For callables, we use the function name and module.
    """
    if hasattr(return_type, "model_json_schema"):
        # Pydantic model - use its schema
        schema_str = str(return_type.model_json_schema())
        # include llm_config if present (not part of schema but affects LLM behaviour)
        if hasattr(return_type, "llm_config"):
            llm_config = return_type.llm_config
            if llm_config is not None:
                schema_str += str(llm_config.model_dump())
        # include YAML definition hash if present (catches any YAML changes)
        if hasattr(return_type, "_yaml_definition_hash"):
            schema_str += return_type._yaml_definition_hash
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
cache_size_limit = get_cache_size_limit()
memory = None


def _get_cache_size(cache_path: Path) -> int:
    """Calculate total size of cache directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(cache_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                # Skip if file doesn't exist (race condition)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.debug(f"Error calculating cache size: {e}")
    return total_size


def _reduce_cache_size_async(memory_obj, size_limit):
    """Reduce cache size in a background thread to avoid blocking module import."""
    try:
        # Get current cache size
        cache_path = Path(memory_obj.location)
        current_size = _get_cache_size(cache_path)

        # Only cleanup if cache has reached the limit
        if current_size >= size_limit:
            memory_obj.reduce_size(bytes_limit=size_limit)
            logger.info(f"Cache size reduced to {size_limit / (1024**3):.1f} GB limit")
        else:
            logger.debug(
                f"Cache size {current_size / (1024**3):.1f} GB within "
                f"{size_limit / (1024**3):.1f} GB limit, cleanup not needed"
            )
    except Exception as e:
        logger.warning(f"Failed to reduce cache size: {e}")


if cache_dir is not None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        memory = Memory(location=str(cache_dir), verbose=0)
        if cache_size_limit:
            # Reduce cache to fit within size limit in background thread
            thread = threading.Thread(
                target=_reduce_cache_size_async,
                args=(memory, cache_size_limit),
                daemon=True,
            )
            thread.start()
            logger.info(
                f"Cache initialized at {cache_dir} with {cache_size_limit / (1024**3):.1f} GB limit"
            )
        else:
            logger.info(f"Cache initialized at {cache_dir} with no size limit")
    except (PermissionError, OSError) as e:
        logger.warning(
            "Failed to create cache directory at ~/.struckdown/cache trying local directory cacheing."
        )
        try:
            fallback_dir = Path(".struckdown/cache")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            memory = Memory(location=str(fallback_dir), verbose=0)
            if cache_size_limit:
                thread = threading.Thread(
                    target=_reduce_cache_size_async,
                    args=(memory, cache_size_limit),
                    daemon=True,
                )
                thread.start()
        except (PermissionError, OSError):
            logger.warning(
                "Failed to create local cache directory .struckdown/cache. Caching is disabled."
            )
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
    # clear embedding cache (uses diskcache, separate from joblib cache)
    from .embedding_cache import clear_embedding_cache

    clear_embedding_cache()

    # clear joblib cache
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
