"""Tests for the prompt cache module."""

import uuid
from pathlib import Path

import pytest

from struckdown.playground import prompt_cache


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Set up a temporary cache directory for tests."""
    cache_dir = tmp_path / "prompts"
    cache_dir.mkdir()
    monkeypatch.setattr(prompt_cache, "PROMPT_CACHE_DIR", cache_dir)
    return cache_dir


class TestValidatePromptId:
    """Tests for UUID validation."""

    def test_valid_uuid(self):
        """Valid UUID passes validation."""
        valid_id = str(uuid.uuid4())
        assert prompt_cache.validate_prompt_id(valid_id) is True

    def test_invalid_uuid(self):
        """Invalid strings fail validation."""
        assert prompt_cache.validate_prompt_id("not-a-uuid") is False
        assert prompt_cache.validate_prompt_id("") is False
        assert prompt_cache.validate_prompt_id("../../../etc/passwd") is False

    def test_path_traversal_blocked(self):
        """Path traversal attempts fail validation."""
        assert prompt_cache.validate_prompt_id("../test") is False
        assert prompt_cache.validate_prompt_id("..%2F..%2Fetc%2Fpasswd") is False


class TestStorePrompt:
    """Tests for storing prompts."""

    def test_store_and_retrieve(self, temp_cache_dir):
        """Store and retrieve prompt text."""
        prompt_id = str(uuid.uuid4())
        text = "Tell me a joke: [[joke]]"

        prompt_cache.store_prompt(prompt_id, text)

        # File should exist on disk
        cache_file = temp_cache_dir / f"{prompt_id}.txt"
        assert cache_file.exists()

        # Content should match
        assert cache_file.read_text(encoding="utf-8") == text

        # Retrieve it
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text

    def test_store_invalid_uuid_raises(self, temp_cache_dir):
        """Storing with invalid UUID raises ValueError."""
        with pytest.raises(ValueError):
            prompt_cache.store_prompt("invalid", "test prompt")

    def test_atomic_write(self, temp_cache_dir):
        """Verify atomic write (no .tmp files left behind)."""
        prompt_id = str(uuid.uuid4())
        prompt_cache.store_prompt(prompt_id, "test prompt")

        # No .tmp files should exist
        tmp_files = list(temp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_store_unicode(self, temp_cache_dir):
        """Store and retrieve unicode text."""
        prompt_id = str(uuid.uuid4())
        text = "Tell me about \u4e2d\u6587 and \u65e5\u672c\u8a9e: [[answer]]"

        prompt_cache.store_prompt(prompt_id, text)
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text

    def test_store_multiline(self, temp_cache_dir):
        """Store and retrieve multiline prompts."""
        prompt_id = str(uuid.uuid4())
        text = """<system>
You are a helpful assistant.
</system>

Tell me a joke about {{topic}}.

[[joke]]
"""
        prompt_cache.store_prompt(prompt_id, text)
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text


class TestGetPrompt:
    """Tests for retrieving prompts."""

    def test_get_nonexistent_raises(self, temp_cache_dir):
        """Getting nonexistent prompt raises FileNotFoundError."""
        prompt_id = str(uuid.uuid4())
        with pytest.raises(FileNotFoundError):
            prompt_cache.get_prompt(prompt_id)

    def test_get_invalid_uuid_raises(self, temp_cache_dir):
        """Getting with invalid UUID raises ValueError."""
        with pytest.raises(ValueError):
            prompt_cache.get_prompt("invalid")


class TestPromptExists:
    """Tests for checking prompt existence."""

    def test_exists_true(self, temp_cache_dir):
        """Existing prompt returns True."""
        prompt_id = str(uuid.uuid4())
        prompt_cache.store_prompt(prompt_id, "test")
        assert prompt_cache.prompt_exists(prompt_id) is True

    def test_exists_false(self, temp_cache_dir):
        """Nonexistent prompt returns False."""
        prompt_id = str(uuid.uuid4())
        assert prompt_cache.prompt_exists(prompt_id) is False

    def test_exists_invalid_uuid(self, temp_cache_dir):
        """Invalid UUID returns False (not exception)."""
        assert prompt_cache.prompt_exists("invalid") is False


class TestGetCacheStats:
    """Tests for cache statistics."""

    def test_empty_cache(self, temp_cache_dir):
        """Empty cache returns zero stats."""
        stats = prompt_cache.get_cache_stats()
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0

    def test_cache_with_files(self, temp_cache_dir):
        """Cache with files returns correct stats."""
        # Store a few prompts
        for i in range(3):
            prompt_cache.store_prompt(str(uuid.uuid4()), f"test prompt {i}" * 100)

        stats = prompt_cache.get_cache_stats()
        assert stats["file_count"] == 3
        assert stats["total_size_mb"] >= 0
