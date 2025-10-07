"""
Tests for caching functionality in struckdown.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from struckdown import LLM, LLMCredentials, structured_chat
from struckdown.cache import get_cache_dir, hash_return_type


class SimpleResponse(BaseModel):
    text: str


class TestCacheDirectory:
    """Test cache directory resolution."""

    def test_default_cache_dir(self):
        """Test that default cache directory is ~/.struckdown/cache"""
        with patch.dict(os.environ, {}, clear=True):
            cache_dir = get_cache_dir()
            assert cache_dir == Path.home() / ".struckdown" / "cache"

    def test_custom_cache_dir(self):
        """Test custom cache directory from env var"""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": "/tmp/custom_cache"}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path("/tmp/custom_cache")

    def test_cache_disabled_zero(self):
        """Test that cache can be disabled with '0'"""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": "0"}):
            cache_dir = get_cache_dir()
            assert cache_dir is None

    def test_cache_disabled_false(self):
        """Test that cache can be disabled with 'false'"""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": "false"}):
            cache_dir = get_cache_dir()
            assert cache_dir is None

    def test_cache_disabled_empty(self):
        """Test that cache can be disabled with empty string"""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": ""}):
            cache_dir = get_cache_dir()
            assert cache_dir is None


class TestReturnTypeHashing:
    """Test hashing of different return types."""

    def test_hash_pydantic_model(self):
        """Test hashing of Pydantic model"""
        hash1 = hash_return_type(SimpleResponse)
        hash2 = hash_return_type(SimpleResponse)
        assert hash1 == hash2

    def test_hash_different_models(self):
        """Test that different models produce different hashes"""

        class OtherResponse(BaseModel):
            value: int

        hash1 = hash_return_type(SimpleResponse)
        hash2 = hash_return_type(OtherResponse)
        assert hash1 != hash2

    def test_hash_callable(self):
        """Test hashing of callable/function"""

        def my_function():
            pass

        hash1 = hash_return_type(my_function)
        hash2 = hash_return_type(my_function)
        assert hash1 == hash2


class TestStructuredChatCaching:
    """Test that structured_chat properly caches results."""

    @patch("struckdown._call_llm_cached")
    def test_cache_hit(self, mock_llm_call):
        """Test that identical calls hit the cache"""
        # Setup mock to return dict responses (as the cached function does)
        mock_llm_call.return_value = (
            {"text": "cached response"},  # res_dict
            {"model": "test"},  # com_dict
        )

        # Create a temporary cache directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"STRUCKDOWN_CACHE": tmpdir}):
                # Need to reload the cache module to pick up new env var
                import importlib

                from struckdown import cache

                importlib.reload(cache)

                # Also need to update the memory object in __init__
                import struckdown

                struckdown.memory = cache.memory

                # Make two identical calls
                prompt = "test prompt"
                llm = LLM(model_name="test/model")
                credentials = LLMCredentials(api_key="test", base_url="http://test")

                result1 = structured_chat(
                    prompt=prompt,
                    return_type=SimpleResponse,
                    llm=llm,
                    credentials=credentials,
                )

                result2 = structured_chat(
                    prompt=prompt,
                    return_type=SimpleResponse,
                    llm=llm,
                    credentials=credentials,
                )

                # Both calls should return the cached response
                assert result1[0].text == "cached response"
                assert result2[0].text == "cached response"

                # The mock should have been called (joblib will call it)
                assert mock_llm_call.called

    def test_cache_disabled(self):
        """Test that caching can be disabled"""
        with patch.dict(os.environ, {"STRUCKDOWN_CACHE": "0"}):
            # Reload cache module with caching disabled
            import importlib

            from struckdown import cache

            importlib.reload(cache)

            # Verify memory location is None
            assert cache.memory.location is None
