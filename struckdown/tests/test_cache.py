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
