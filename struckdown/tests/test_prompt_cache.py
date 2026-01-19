"""Tests for the prompt cache module."""

import os
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


class TestHashContent:
    """Tests for content hashing."""

    def test_deterministic_hash(self):
        """Same content always produces same hash."""
        text = "Tell me a joke: [[joke]]"
        hash1 = prompt_cache.hash_content(text)
        hash2 = prompt_cache.hash_content(text)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content produces different hashes."""
        hash1 = prompt_cache.hash_content("Tell me a joke")
        hash2 = prompt_cache.hash_content("Tell me a story")
        assert hash1 != hash2

    def test_hash_length(self):
        """Hash is expected length."""
        text = "test"
        h = prompt_cache.hash_content(text)
        assert len(h) == prompt_cache.HASH_LENGTH


class TestValidatePromptId:
    """Tests for hash validation."""

    def test_valid_hash(self):
        """Valid hex hash passes validation."""
        # Generate a real hash
        valid_id = prompt_cache.hash_content("test")
        assert prompt_cache.validate_prompt_id(valid_id) is True

    def test_invalid_hash(self):
        """Invalid strings fail validation."""
        assert prompt_cache.validate_prompt_id("not-a-hash") is False
        assert prompt_cache.validate_prompt_id("") is False
        assert prompt_cache.validate_prompt_id("../../../etc/passwd") is False

    def test_wrong_length(self):
        """Wrong length fails validation."""
        assert prompt_cache.validate_prompt_id("abc123") is False  # too short
        assert prompt_cache.validate_prompt_id("a" * 100) is False  # too long

    def test_path_traversal_blocked(self):
        """Path traversal attempts fail validation."""
        assert prompt_cache.validate_prompt_id("../test") is False
        assert prompt_cache.validate_prompt_id("..%2F..%2Fetc%2Fpasswd") is False

    def test_uppercase_rejected(self):
        """Uppercase hex is rejected (we use lowercase)."""
        # Create valid hash then uppercase it
        valid = prompt_cache.hash_content("test")
        assert prompt_cache.validate_prompt_id(valid.upper()) is False


class TestStorePrompt:
    """Tests for storing prompts."""

    def test_store_and_retrieve(self, temp_cache_dir):
        """Store and retrieve prompt text."""
        text = "Tell me a joke: [[joke]]"

        prompt_id, warning = prompt_cache.store_prompt(text)
        assert warning is None

        # File should exist on disk
        cache_file = temp_cache_dir / f"{prompt_id}.txt"
        assert cache_file.exists()

        # Content should match
        assert cache_file.read_text(encoding="utf-8") == text

        # Retrieve it
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text

    def test_store_returns_hash(self, temp_cache_dir):
        """store_prompt returns valid content hash."""
        text = "test prompt"
        prompt_id, warning = prompt_cache.store_prompt(text)

        # Should be valid hash
        assert prompt_cache.validate_prompt_id(prompt_id)

        # Should match hash_content
        assert prompt_id == prompt_cache.hash_content(text)

    def test_idempotent_store(self, temp_cache_dir):
        """Storing same content twice returns same ID."""
        text = "same content"

        id1, _ = prompt_cache.store_prompt(text)
        id2, _ = prompt_cache.store_prompt(text)

        assert id1 == id2

        # Only one file should exist
        files = list(temp_cache_dir.glob("*.txt"))
        assert len(files) == 1

    def test_atomic_write(self, temp_cache_dir):
        """Verify atomic write (no .tmp files left behind)."""
        prompt_cache.store_prompt("test prompt")

        # No .tmp files should exist
        tmp_files = list(temp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_store_unicode(self, temp_cache_dir):
        """Store and retrieve unicode text."""
        text = "Tell me about \u4e2d\u6587 and \u65e5\u672c\u8a9e: [[answer]]"

        prompt_id, _ = prompt_cache.store_prompt(text)
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text

    def test_store_multiline(self, temp_cache_dir):
        """Store and retrieve multiline prompts."""
        text = """<system>
You are a helpful assistant.
</system>

Tell me a joke about {{topic}}.

[[joke]]
"""
        prompt_id, _ = prompt_cache.store_prompt(text)
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert retrieved == text

    def test_file_permissions(self, temp_cache_dir):
        """Stored files have restricted permissions."""
        prompt_id, _ = prompt_cache.store_prompt("test")
        cache_file = temp_cache_dir / f"{prompt_id}.txt"

        # Check permissions are 0600 (owner read/write only)
        mode = cache_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_truncates_large_prompt(self, temp_cache_dir, monkeypatch):
        """Large prompts are truncated with warning."""
        # Set small limit for testing
        monkeypatch.setattr(prompt_cache, "MAX_PROMPT_SIZE", 100)

        text = "x" * 200  # Exceeds limit

        prompt_id, warning = prompt_cache.store_prompt(text)
        assert warning is not None
        assert "truncated" in warning.lower()

        # Retrieved content should be truncated
        retrieved = prompt_cache.get_prompt(prompt_id)
        assert len(retrieved) == 100


class TestGetPrompt:
    """Tests for retrieving prompts."""

    def test_get_nonexistent_raises(self, temp_cache_dir):
        """Getting nonexistent prompt raises FileNotFoundError."""
        # Create a valid hash that doesn't exist
        fake_id = "a" * prompt_cache.HASH_LENGTH
        with pytest.raises(FileNotFoundError):
            prompt_cache.get_prompt(fake_id)

    def test_get_invalid_hash_raises(self, temp_cache_dir):
        """Getting with invalid hash raises ValueError."""
        with pytest.raises(ValueError):
            prompt_cache.get_prompt("invalid")


class TestPromptExists:
    """Tests for checking prompt existence."""

    def test_exists_true(self, temp_cache_dir):
        """Existing prompt returns True."""
        prompt_id, _ = prompt_cache.store_prompt("test")
        assert prompt_cache.prompt_exists(prompt_id) is True

    def test_exists_false(self, temp_cache_dir):
        """Nonexistent prompt returns False."""
        fake_id = "b" * prompt_cache.HASH_LENGTH
        assert prompt_cache.prompt_exists(fake_id) is False

    def test_exists_invalid_hash(self, temp_cache_dir):
        """Invalid hash returns False (not exception)."""
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
        # Store a few prompts with different content
        for i in range(3):
            prompt_cache.store_prompt(f"test prompt {i}" * 100)

        stats = prompt_cache.get_cache_stats()
        assert stats["file_count"] == 3
        assert stats["total_size_mb"] >= 0
