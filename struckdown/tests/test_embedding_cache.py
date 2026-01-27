"""
Tests for embedding cache functionality.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from struckdown.embedding_cache import (
    _make_cache_key,
    clear_embedding_cache,
    get_cached_embeddings,
    store_embeddings,
)


class TestCacheKeyDeterminism:
    """Test that cache keys are deterministic and correctly formed."""

    def test_same_input_same_key(self):
        """Same text, model, dimensions should produce same key."""
        key1 = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        key2 = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        assert key1 == key2

    def test_different_text_different_key(self):
        """Different text should produce different key."""
        key1 = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        key2 = _make_cache_key("goodbye world", "text-embedding-3-large", 3072)
        assert key1 != key2

    def test_different_model_different_key(self):
        """Different model should produce different key."""
        key1 = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        key2 = _make_cache_key("hello world", "text-embedding-3-small", 3072)
        assert key1 != key2

    def test_different_dimensions_different_key(self):
        """Different dimensions should produce different key."""
        key1 = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        key2 = _make_cache_key("hello world", "text-embedding-3-large", 1536)
        assert key1 != key2

    def test_none_dimensions_produces_valid_key(self):
        """None dimensions should produce a valid key."""
        key = _make_cache_key("hello world", "text-embedding-3-large", None)
        assert "none" in key
        assert "text-embedding-3-large" in key

    def test_key_format(self):
        """Key should contain model, dimensions, and text hash."""
        key = _make_cache_key("hello world", "text-embedding-3-large", 3072)
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "text-embedding-3-large"
        assert parts[1] == "3072"
        assert len(parts[2]) == 32  # sha256[:32] = 128 bits


class TestCacheStoreRetrieve:
    """Test storing and retrieving embeddings from cache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "embeddings")
            with patch.dict(os.environ, {"STRUCKDOWN_CACHE": tmpdir}):
                # reset global cache
                import struckdown.embedding_cache as ec

                ec._embedding_cache = None
                yield cache_path
                # cleanup
                ec._embedding_cache = None

    def test_store_and_retrieve_single(self, temp_cache_dir):
        """Store and retrieve a single embedding."""
        texts = ["hello world"]
        embeddings = [[0.1, 0.2, 0.3]]

        store_embeddings(texts, embeddings, "test-model", 3)
        cached, missing = get_cached_embeddings(texts, "test-model", 3)

        assert len(cached) == 1
        assert len(missing) == 0
        assert cached[0] == embeddings[0]

    def test_store_and_retrieve_multiple(self, temp_cache_dir):
        """Store and retrieve multiple embeddings."""
        texts = ["hello", "world", "test"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        store_embeddings(texts, embeddings, "test-model", 2)
        cached, missing = get_cached_embeddings(texts, "test-model", 2)

        assert len(cached) == 3
        assert len(missing) == 0
        for i in range(3):
            assert cached[i] == embeddings[i]

    def test_partial_cache_hit(self, temp_cache_dir):
        """Test that partial cache hits work correctly."""
        # store first batch
        texts1 = ["hello", "world"]
        embeddings1 = [[0.1, 0.2], [0.3, 0.4]]
        store_embeddings(texts1, embeddings1, "test-model", 2)

        # query with overlap
        texts2 = ["world", "new"]
        cached, missing = get_cached_embeddings(texts2, "test-model", 2)

        assert len(cached) == 1  # "world" is cached
        assert 0 in cached  # "world" is at index 0 in texts2
        assert cached[0] == [0.3, 0.4]
        assert len(missing) == 1
        assert missing[0] == (1, "new")  # "new" is missing

    def test_different_model_no_hit(self, temp_cache_dir):
        """Embeddings cached for one model shouldn't hit for another."""
        texts = ["hello"]
        embeddings = [[0.1, 0.2]]

        store_embeddings(texts, embeddings, "model-a", 2)
        cached, missing = get_cached_embeddings(texts, "model-b", 2)

        assert len(cached) == 0
        assert len(missing) == 1


class TestCacheDisabled:
    """Test behaviour when caching is disabled."""

    def test_disabled_cache_returns_all_missing(self):
        """When cache is disabled, all texts should be marked as missing."""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": "0"}):
            import struckdown.embedding_cache as ec

            ec._embedding_cache = None  # reset

            texts = ["hello", "world"]
            cached, missing = get_cached_embeddings(texts, "test-model", 3072)

            assert len(cached) == 0
            assert len(missing) == 2
            assert missing[0] == (0, "hello")
            assert missing[1] == (1, "world")


class TestClearCache:
    """Test cache clearing functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"STRUCKDOWN_CACHE": tmpdir}):
                import struckdown.embedding_cache as ec

                ec._embedding_cache = None
                yield tmpdir
                ec._embedding_cache = None

    def test_clear_removes_embeddings(self, temp_cache_dir):
        """Clearing cache should remove all stored embeddings."""
        texts = ["hello", "world"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        store_embeddings(texts, embeddings, "test-model", 2)

        # verify stored
        cached, _ = get_cached_embeddings(texts, "test-model", 2)
        assert len(cached) == 2

        # clear
        clear_embedding_cache()

        # reset to reinitialise cache
        import struckdown.embedding_cache as ec

        ec._embedding_cache = None

        # verify cleared
        cached, missing = get_cached_embeddings(texts, "test-model", 2)
        assert len(cached) == 0
        assert len(missing) == 2
