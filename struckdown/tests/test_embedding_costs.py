"""
Tests for embedding cost tracking functionality.
"""

import numpy as np
import pickle
import pytest

from struckdown.llm import EmbeddingResult, EmbeddingResultList


class TestEmbeddingResult:
    """Test EmbeddingResult numpy array subclass."""

    def test_creates_array_with_metadata(self):
        """EmbeddingResult should store cost metadata."""
        emb = EmbeddingResult([0.1, 0.2, 0.3], cost=0.001, tokens=5, model="test-model", cached=False)

        assert emb.cost == 0.001
        assert emb.tokens == 5
        assert emb.model == "test-model"
        assert emb.cached is False

    def test_behaves_like_ndarray(self):
        """EmbeddingResult should support numpy operations."""
        emb1 = EmbeddingResult([1.0, 0.0, 0.0], cost=0.001, tokens=3, model="test", cached=False)
        emb2 = EmbeddingResult([0.0, 1.0, 0.0], cost=0.001, tokens=3, model="test", cached=False)

        # dot product
        dot = np.dot(emb1, emb2)
        assert dot == 0.0

        # cosine similarity (orthogonal vectors)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        assert norm1 == 1.0
        assert norm2 == 1.0

    def test_array_operations_preserve_type(self):
        """Operations should preserve EmbeddingResult type where sensible."""
        emb = EmbeddingResult([1.0, 2.0, 3.0], cost=0.001, tokens=3, model="test", cached=False)

        # slicing preserves type
        sliced = emb[:2]
        assert isinstance(sliced, EmbeddingResult)

    def test_default_values(self):
        """Default metadata values should be sensible."""
        emb = EmbeddingResult([0.1, 0.2])

        assert emb.cost is None
        assert emb.tokens == 0
        assert emb.model == ""
        assert emb.cached is False

    def test_none_cost_for_unknown(self):
        """Cost should be None when unknown, not 0."""
        emb = EmbeddingResult([0.1, 0.2], cost=None, tokens=5, model="test", cached=False)
        assert emb.cost is None

    def test_cached_embedding_has_zero_cost(self):
        """Cached embeddings should report zero cost."""
        emb = EmbeddingResult([0.1, 0.2], cost=0.0, tokens=0, model="test", cached=True)

        assert emb.cached is True
        assert emb.cost == 0.0

    def test_pickle_roundtrip(self):
        """EmbeddingResult should pickle correctly with metadata."""
        emb = EmbeddingResult([0.1, 0.2, 0.3], cost=0.002, tokens=10, model="test-model", cached=True)

        pickled = pickle.dumps(emb)
        restored = pickle.loads(pickled)

        assert np.array_equal(restored, emb)
        assert restored.cost == emb.cost
        assert restored.tokens == emb.tokens
        assert restored.model == emb.model
        assert restored.cached == emb.cached


class TestEmbeddingResultList:
    """Test EmbeddingResultList container."""

    def test_aggregate_costs(self):
        """List should aggregate costs from items."""
        emb1 = EmbeddingResult([0.1], cost=0.001, tokens=5, model="test", cached=False)
        emb2 = EmbeddingResult([0.2], cost=0.002, tokens=10, model="test", cached=False)
        emb3 = EmbeddingResult([0.3], cost=0.0, tokens=0, model="test", cached=True)

        results = EmbeddingResultList([emb1, emb2, emb3], model="test")

        assert results.total_cost == pytest.approx(0.003)
        assert results.total_tokens == 15
        assert results.cached_count == 1
        assert results.fresh_count == 2
        assert results.fresh_cost == pytest.approx(0.003)

    def test_behaves_like_list(self):
        """List should support standard list operations."""
        emb1 = EmbeddingResult([0.1], cost=0.001, tokens=5, model="test", cached=False)
        emb2 = EmbeddingResult([0.2], cost=0.002, tokens=10, model="test", cached=False)

        results = EmbeddingResultList([emb1, emb2], model="test")

        assert len(results) == 2
        assert results[0] is emb1
        assert results[1] is emb2

        # iteration
        items = list(results)
        assert items == [emb1, emb2]

    def test_empty_list(self):
        """Empty list should have zero costs."""
        results = EmbeddingResultList([], model="test")

        assert results.total_cost == 0.0
        assert results.total_tokens == 0
        assert results.cached_count == 0
        assert results.fresh_count == 0

    def test_all_cached(self):
        """List of all cached embeddings should have zero cost."""
        emb1 = EmbeddingResult([0.1], cost=0.0, tokens=0, model="test", cached=True)
        emb2 = EmbeddingResult([0.2], cost=0.0, tokens=0, model="test", cached=True)

        results = EmbeddingResultList([emb1, emb2], model="test")

        assert results.total_cost == 0.0
        assert results.fresh_cost == 0.0
        assert results.cached_count == 2
        assert results.fresh_count == 0

    def test_model_property(self):
        """List should expose model name."""
        results = EmbeddingResultList([], model="text-embedding-3-large")
        assert results.model == "text-embedding-3-large"

    def test_unknown_cost_returns_none(self):
        """List with unknown costs should return None for total_cost."""
        emb1 = EmbeddingResult([0.1], cost=0.001, tokens=5, model="test", cached=False)
        emb2 = EmbeddingResult([0.2], cost=None, tokens=10, model="test", cached=False)

        results = EmbeddingResultList([emb1, emb2], model="test")

        assert results.has_unknown_costs is True
        assert results.total_cost is None
        assert results.fresh_cost is None
        assert results.total_tokens == 15

    def test_cached_with_unknown_fresh_cost(self):
        """Cached items don't count toward unknown costs."""
        emb1 = EmbeddingResult([0.1], cost=0.0, tokens=0, model="test", cached=True)
        emb2 = EmbeddingResult([0.2], cost=0.001, tokens=5, model="test", cached=False)

        results = EmbeddingResultList([emb1, emb2], model="test")

        assert results.has_unknown_costs is False
        assert results.total_cost == pytest.approx(0.001)


class TestBackwardsCompatibility:
    """Test that existing code continues to work."""

    def test_iteration_still_works(self):
        """Iterating over results should still yield arrays."""
        emb1 = EmbeddingResult([0.1, 0.2], cost=0.001, tokens=5, model="test", cached=False)
        emb2 = EmbeddingResult([0.3, 0.4], cost=0.001, tokens=5, model="test", cached=False)
        results = EmbeddingResultList([emb1, emb2], model="test")

        # this pattern should still work
        for emb in results:
            assert len(emb) == 2
            assert isinstance(emb, np.ndarray)

    def test_indexing_returns_array(self):
        """Indexing should return usable arrays."""
        emb = EmbeddingResult([0.1, 0.2, 0.3], cost=0.001, tokens=5, model="test", cached=False)
        results = EmbeddingResultList([emb], model="test")

        # direct use as array
        arr = results[0]
        assert np.dot(arr, arr) == pytest.approx(0.14)

    def test_numpy_operations_on_list_items(self):
        """Items should work with numpy functions."""
        emb1 = EmbeddingResult([1.0, 0.0], cost=0.001, tokens=5, model="test", cached=False)
        emb2 = EmbeddingResult([0.0, 1.0], cost=0.001, tokens=5, model="test", cached=False)
        results = EmbeddingResultList([emb1, emb2], model="test")

        # stack into matrix
        matrix = np.stack(list(results))
        assert matrix.shape == (2, 2)

        # cosine similarity matrix
        similarity = np.dot(matrix, matrix.T)
        assert similarity.shape == (2, 2)
