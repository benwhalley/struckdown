"""
Per-string embedding cache using diskcache.

Caches individual text embeddings to avoid redundant API calls when
the same text appears in different batches.

Cache location: ~/.struckdown/cache/embeddings/
Key format: {model}:{dimensions}:{sha256(text)[:16]}
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from decouple import config as env_config

logger = logging.getLogger(__name__)

# lazy-initialised cache instance
_embedding_cache = None


def _get_embedding_cache_dir() -> Optional[Path]:
    """Get the embedding cache directory."""
    cache_setting = env_config("STRUCKDOWN_CACHE", default="~/.struckdown/cache")

    if cache_setting.lower() in ("0", "false", ""):
        return None

    return Path(cache_setting).expanduser() / "embeddings"


def _get_embedding_cache():
    """Get or create the embedding cache (lazy initialisation)."""
    global _embedding_cache

    if _embedding_cache is not None:
        return _embedding_cache

    cache_dir = _get_embedding_cache_dir()
    if cache_dir is None:
        return None

    from diskcache import FanoutCache

    cache_dir.mkdir(parents=True, exist_ok=True)

    # FanoutCache provides thread-safe access with multiple shards
    _embedding_cache = FanoutCache(
        directory=str(cache_dir),
        shards=8,
        timeout=1,  # seconds to wait for lock
        size_limit=5 * 1024 * 1024 * 1024,  # 5GB
        eviction_policy="least-recently-used",
    )

    logger.debug(f"Embedding cache initialised at {cache_dir}")
    return _embedding_cache


def _make_cache_key(text: str, model: str, dimensions: Optional[int]) -> str:
    """Create a deterministic cache key for an embedding."""
    # 32 hex chars = 128 bits, collision-safe for billions of entries
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    dims_str = str(dimensions) if dimensions is not None else "none"
    return f"{model}:{dims_str}:{text_hash}"


def get_cached_embeddings(
    texts: List[str],
    model: str,
    dimensions: Optional[int],
) -> Tuple[Dict[int, List[float]], List[Tuple[int, str]]]:
    """
    Check cache for existing embeddings.

    Args:
        texts: List of texts to check
        model: Model name used for embeddings
        dimensions: Embedding dimensions (or None)

    Returns:
        Tuple of:
        - cached: Dict mapping index -> embedding for texts found in cache
        - missing: List of (index, text) tuples for texts not in cache
    """
    cache = _get_embedding_cache()

    cached: Dict[int, List[float]] = {}
    missing: List[Tuple[int, str]] = []

    if cache is None:
        # caching disabled, all texts are "missing"
        return {}, [(i, text) for i, text in enumerate(texts)]

    for idx, text in enumerate(texts):
        key = _make_cache_key(text, model, dimensions)
        embedding = cache.get(key)

        if embedding is not None:
            cached[idx] = embedding
        else:
            missing.append((idx, text))

    if cached:
        logger.debug(f"Embedding cache: {len(cached)} hits, {len(missing)} misses")

    return cached, missing


def store_embeddings(
    texts: List[str],
    embeddings: List[List[float]],
    model: str,
    dimensions: Optional[int],
) -> None:
    """
    Store embeddings in cache.

    Args:
        texts: List of texts that were embedded
        embeddings: Corresponding embedding vectors
        model: Model name used
        dimensions: Embedding dimensions (or None)
    """
    cache = _get_embedding_cache()

    if cache is None:
        return

    for text, embedding in zip(texts, embeddings):
        key = _make_cache_key(text, model, dimensions)
        cache.set(key, embedding)

    logger.debug(f"Stored {len(texts)} embeddings in cache")


def clear_embedding_cache() -> None:
    """Clear all cached embeddings."""
    global _embedding_cache

    cache_dir = _get_embedding_cache_dir()
    if cache_dir is None:
        logger.info("Embedding caching is disabled, nothing to clear")
        return

    # close existing cache if open
    if _embedding_cache is not None:
        _embedding_cache.close()
        _embedding_cache = None

    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        logger.info(f"Cleared embedding cache directory: {cache_dir}")
    else:
        logger.info(f"Embedding cache directory does not exist: {cache_dir}")


def _make_pair_cache_key(text_a: str, text_b: str, model: str) -> str:
    """Create a deterministic cache key for a cross-encoder pair score."""
    # hash both texts together to form a unique key for the pair
    combined = f"{text_a}\x00{text_b}"  # null byte separator
    pair_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:32]
    return f"pair:{model}:{pair_hash}"


def get_cached_pair_scores(
    pairs: List[Tuple[str, str]],
    model: str,
) -> Tuple[Dict[int, float], List[Tuple[int, Tuple[str, str]]]]:
    """
    Check cache for existing cross-encoder pair scores.

    Args:
        pairs: List of (text_a, text_b) tuples to check
        model: Model name used for scoring

    Returns:
        Tuple of:
        - cached: Dict mapping index -> score for pairs found in cache
        - missing: List of (index, pair) tuples for pairs not in cache
    """
    cache = _get_embedding_cache()

    cached: Dict[int, float] = {}
    missing: List[Tuple[int, Tuple[str, str]]] = []

    if cache is None:
        # caching disabled, all pairs are "missing"
        return {}, [(i, pair) for i, pair in enumerate(pairs)]

    for idx, (text_a, text_b) in enumerate(pairs):
        key = _make_pair_cache_key(text_a, text_b, model)
        score = cache.get(key)

        if score is not None:
            cached[idx] = score
        else:
            missing.append((idx, (text_a, text_b)))

    if cached:
        logger.debug(f"Pair score cache: {len(cached)} hits, {len(missing)} misses")

    return cached, missing


def store_pair_scores(
    pairs: List[Tuple[str, str]],
    scores: List[float],
    model: str,
) -> None:
    """
    Store cross-encoder pair scores in cache.

    Args:
        pairs: List of (text_a, text_b) tuples that were scored
        scores: Corresponding scores
        model: Model name used
    """
    cache = _get_embedding_cache()

    if cache is None:
        return

    for (text_a, text_b), score in zip(pairs, scores):
        key = _make_pair_cache_key(text_a, text_b, model)
        cache.set(key, score)

    logger.debug(f"Stored {len(pairs)} pair scores in cache")
