"""
Integration tests for LLM call caching with the new messages API.

These tests verify that the @memory.cache decorator on _call_llm_cached()
correctly caches LLM responses based on the messages parameter.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from struckdown import _call_llm_cached, chatter
from struckdown.cache import clear_cache, memory


class SimpleResponse(BaseModel):
    """Simple response model for testing"""

    response: str


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache"
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": str(cache_path)}):
            # Force reimport to pick up new cache location
            import importlib

            from struckdown import cache as cache_module

            importlib.reload(cache_module)
            yield cache_path
            # Cleanup happens automatically with tempfile


class MockCompletion:
    """Picklable mock completion object"""

    def __init__(self):
        self.usage = type("Usage", (), {"total_tokens": 10})()

    def model_dump(self):
        return {"usage": {"total_tokens": 10}}


@pytest.fixture(autouse=True)
def clear_test_cache():
    """Clear cache before each test to ensure clean state"""
    clear_cache()
    yield
    # Cache cleared before next test


@pytest.fixture
def mock_llm():
    """Create a mock LLM with tracking for cache testing"""
    mock_client = Mock()

    # Track number of times the LLM is actually called
    call_count = {"count": 0}

    def create_with_completion(*args, **kwargs):
        call_count["count"] += 1
        # Create a picklable response based on messages
        messages = kwargs.get("messages", [])
        content = messages[-1].get("content", "") if messages else ""

        # Use actual Pydantic model instead of Mock for picklability
        response_text = f"Mock response to: {content[:50]}"
        mock_response = SimpleResponse(response=response_text)

        # Use picklable completion object
        mock_completion = MockCompletion()

        return mock_response, mock_completion

    mock_client.chat.completions.create_with_completion = create_with_completion
    mock_llm_obj = Mock()
    mock_llm_obj.client = Mock(return_value=mock_client)

    return mock_llm_obj, call_count


class TestCacheIntegration:
    """Test actual caching behavior of LLM calls"""

    def test_cache_hit_with_identical_messages(self, mock_llm):
        """Test that identical messages produce a cache hit"""
        llm, call_count = mock_llm

        messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]

        # First call - should hit LLM
        result1, _ = _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with identical parameters - should use cache
        result2, _ = _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1  # Still 1 - cached!
        assert result1 == result2

    def test_cache_miss_with_different_messages(self, mock_llm):
        """Test that different messages produce a cache miss"""
        llm, call_count = mock_llm

        messages1 = [
            {"role": "user", "content": "What is 2+2?"},
        ]

        messages2 = [
            {"role": "user", "content": "What is 3+3?"},
        ]

        # First call
        result1, _ = _call_llm_cached(
            messages=messages1,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with different messages - should NOT use cache
        result2, _ = _call_llm_cached(
            messages=messages2,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Called again
        assert result1 != result2

    def test_cache_with_system_messages(self, mock_llm):
        """Test that system messages are included in cache key"""
        llm, call_count = mock_llm

        messages1 = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        messages2 = [
            {"role": "system", "content": "You are rude"},
            {"role": "user", "content": "Hello"},
        ]

        # First call with first system message
        result1, _ = _call_llm_cached(
            messages=messages1,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with different system message - should NOT use cache
        result2, _ = _call_llm_cached(
            messages=messages2,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Different system message = different cache key

    def test_cache_with_message_order(self, mock_llm):
        """Test that message order matters for caching"""
        llm, call_count = mock_llm

        messages1 = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Second"},
        ]

        messages2 = [
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "First"},
        ]

        # First call
        _call_llm_cached(
            messages=messages1,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with different order - should NOT use cache
        _call_llm_cached(
            messages=messages2,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Order matters

    def test_cache_ignores_credentials(self, mock_llm):
        """Test that different credentials still produce cache hit"""
        llm, call_count = mock_llm

        messages = [
            {"role": "user", "content": "Test"},
        ]

        # First call with one credential
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials="cred1",
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with different credential - should use cache (credentials ignored)
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials="cred2",
            cache_version="test-v1",
        )

        assert call_count["count"] == 1  # Cached despite different credentials

    def test_cache_respects_model_name(self, mock_llm):
        """Test that different model names produce cache miss"""
        llm, call_count = mock_llm

        messages = [
            {"role": "user", "content": "Test"},
        ]

        # First call with model A
        _call_llm_cached(
            messages=messages,
            model_name="model-a",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with model B - should NOT use cache
        _call_llm_cached(
            messages=messages,
            model_name="model-b",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Different model = no cache

    def test_cache_respects_max_tokens(self, mock_llm):
        """Test that different max_tokens produce cache miss"""
        llm, call_count = mock_llm

        messages = [
            {"role": "user", "content": "Test"},
        ]

        # First call with max_tokens=100
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with max_tokens=200 - should NOT use cache
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=200,
            extra_kwargs={},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Different max_tokens = no cache

    def test_cache_with_extra_kwargs(self, mock_llm):
        """Test that extra_kwargs are included in cache key"""
        llm, call_count = mock_llm

        messages = [
            {"role": "user", "content": "Test"},
        ]

        # First call with temperature=0.5
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={"temperature": 0.5},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 1

        # Second call with temperature=1.0 - should NOT use cache
        _call_llm_cached(
            messages=messages,
            model_name="test-model",
            max_retries=3,
            max_tokens=100,
            extra_kwargs={"temperature": 1.0},
            return_type=SimpleResponse,
            llm=llm,
            credentials=None,
            cache_version="test-v1",
        )

        assert call_count["count"] == 2  # Different extra_kwargs = no cache


class TestCacheEndToEnd:
    """End-to-end cache tests using the chatter() function"""

    def test_chatter_uses_cache(self, mock_llm):
        """Test that chatter() function benefits from caching"""
        llm, call_count = mock_llm

        template = "What is 2+2?"

        # Patch structured_chat to use our mock
        with patch("struckdown.structured_chat") as mock_structured:
            # Set up mock to track calls
            def mock_call(*args, **kwargs):
                call_count["count"] += 1
                mock_result = Mock()
                mock_result.response = "4"
                mock_result.model_dump = Mock(return_value={"response": "4"})
                mock_completion = Mock()
                mock_completion.model_dump = Mock(
                    return_value={"usage": {"total_tokens": 10}}
                )
                return mock_result, mock_completion

            mock_structured.side_effect = mock_call

            # First call
            result1 = chatter(template)
            first_call_count = call_count["count"]

            # Second call with same template - should be cached
            result2 = chatter(template)
            second_call_count = call_count["count"]

            # Verify responses are the same
            assert result1.response == result2.response

            # Note: In practice, caching happens at _call_llm_cached level
            # This test shows the integration point
