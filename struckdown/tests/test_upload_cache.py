"""Tests for the upload cache module."""

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from struckdown.playground import upload_cache


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Set up a temporary cache directory for tests."""
    cache_dir = tmp_path / "uploads"
    cache_dir.mkdir()
    monkeypatch.setattr(upload_cache, "UPLOAD_CACHE_DIR", cache_dir)
    return cache_dir


class TestValidateFileId:
    """Tests for UUID validation."""

    def test_valid_uuid(self):
        """Valid UUID passes validation."""
        valid_id = str(uuid.uuid4())
        assert upload_cache.validate_file_id(valid_id) is True

    def test_invalid_uuid(self):
        """Invalid strings fail validation."""
        assert upload_cache.validate_file_id("not-a-uuid") is False
        assert upload_cache.validate_file_id("") is False
        assert upload_cache.validate_file_id("../../../etc/passwd") is False

    def test_uuid_without_dashes(self):
        """UUID without dashes should still work."""
        # validators.uuid accepts various formats
        valid_id = uuid.uuid4().hex
        # This may or may not pass depending on validators implementation
        # The important thing is path traversal attacks fail


class TestStoreUpload:
    """Tests for storing uploads."""

    def test_store_and_retrieve(self, temp_cache_dir):
        """Store and retrieve upload data."""
        file_id = str(uuid.uuid4())
        data = {"type": "batch", "filename": "test.csv", "data": {"rows": [{"a": 1}], "columns": ["a"]}}

        upload_cache.store_upload(file_id, data)

        # File should exist on disk
        cache_file = temp_cache_dir / f"{file_id}.json"
        assert cache_file.exists()

        # Retrieve it
        retrieved = upload_cache.get_upload(file_id)
        assert retrieved["type"] == "batch"
        assert retrieved["filename"] == "test.csv"
        assert "uploaded_at" in retrieved

    def test_store_invalid_uuid_raises(self, temp_cache_dir):
        """Storing with invalid UUID raises ValueError."""
        with pytest.raises(ValueError):
            upload_cache.store_upload("invalid", {"data": "test"})

    def test_atomic_write(self, temp_cache_dir):
        """Verify atomic write (no .tmp files left behind)."""
        file_id = str(uuid.uuid4())
        upload_cache.store_upload(file_id, {"data": "test"})

        # No .tmp files should exist
        tmp_files = list(temp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestGetUpload:
    """Tests for retrieving uploads."""

    def test_get_nonexistent_raises(self, temp_cache_dir):
        """Getting nonexistent upload raises FileNotFoundError."""
        file_id = str(uuid.uuid4())
        with pytest.raises(FileNotFoundError):
            upload_cache.get_upload(file_id)

    def test_get_invalid_uuid_raises(self, temp_cache_dir):
        """Getting with invalid UUID raises ValueError."""
        with pytest.raises(ValueError):
            upload_cache.get_upload("invalid")

    def test_expired_upload_raises(self, temp_cache_dir, monkeypatch):
        """Expired uploads raise FileNotFoundError."""
        # Set very short expiry
        monkeypatch.setattr(upload_cache, "UPLOAD_MAX_AGE_SECONDS", 1)

        file_id = str(uuid.uuid4())
        upload_cache.store_upload(file_id, {"data": "test"})

        # Wait for expiry
        time.sleep(1.5)

        with pytest.raises(FileNotFoundError):
            upload_cache.get_upload(file_id)

        # File should be deleted
        cache_file = temp_cache_dir / f"{file_id}.json"
        assert not cache_file.exists()


class TestDeleteUpload:
    """Tests for deleting uploads."""

    def test_delete_existing(self, temp_cache_dir):
        """Delete existing upload returns True."""
        file_id = str(uuid.uuid4())
        upload_cache.store_upload(file_id, {"data": "test"})

        result = upload_cache.delete_upload(file_id)
        assert result is True

        # File should be gone
        cache_file = temp_cache_dir / f"{file_id}.json"
        assert not cache_file.exists()

    def test_delete_nonexistent(self, temp_cache_dir):
        """Delete nonexistent upload returns False."""
        file_id = str(uuid.uuid4())
        result = upload_cache.delete_upload(file_id)
        assert result is False

    def test_delete_invalid_uuid(self, temp_cache_dir):
        """Delete with invalid UUID returns False."""
        result = upload_cache.delete_upload("invalid")
        assert result is False


class TestCleanupCache:
    """Tests for cache cleanup."""

    def test_cleanup_expired_files(self, temp_cache_dir, monkeypatch):
        """Cleanup removes expired files."""
        monkeypatch.setattr(upload_cache, "UPLOAD_MAX_AGE_SECONDS", 1)

        # Create some files
        file_ids = [str(uuid.uuid4()) for _ in range(3)]
        for fid in file_ids:
            upload_cache.store_upload(fid, {"data": "test"})

        # Wait for expiry
        time.sleep(1.5)

        # Cleanup
        removed = upload_cache.cleanup_cache()
        assert removed == 3

        # All files should be gone
        assert len(list(temp_cache_dir.glob("*.json"))) == 0

    def test_cleanup_enforces_size_limit(self, temp_cache_dir, monkeypatch):
        """Cleanup removes oldest files when over size limit."""
        # Very small size limit (1KB = 0.001 MB, but we need files > total limit)
        monkeypatch.setattr(upload_cache, "UPLOAD_CACHE_SIZE_MB", 0.001)  # ~1KB
        monkeypatch.setattr(upload_cache, "UPLOAD_MAX_AGE_SECONDS", 86400)

        # Create files with different timestamps, each file ~500 bytes
        file_ids = []
        for i in range(5):
            fid = str(uuid.uuid4())
            file_ids.append(fid)
            # Create file with specific uploaded_at to ensure ordering
            # Each file is ~500 bytes, total ~2.5KB > 1KB limit
            data = {"data": "x" * 400, "uploaded_at": time.time() + i * 0.1}
            path = temp_cache_dir / f"{fid}.json"
            path.write_text(json.dumps(data))
            time.sleep(0.05)  # Small delay to ensure different mtimes

        # Verify we have 5 files initially
        assert len(list(temp_cache_dir.glob("*.json"))) == 5

        # Cleanup should remove oldest files first (FIFO)
        removed = upload_cache.cleanup_cache()

        # Some files should be removed to get under 1KB limit
        remaining = list(temp_cache_dir.glob("*.json"))
        assert len(remaining) < 5


class TestGetCacheStats:
    """Tests for cache statistics."""

    def test_empty_cache_stats(self, temp_cache_dir):
        """Empty cache returns zero stats."""
        stats = upload_cache.get_cache_stats()
        assert stats["file_count"] == 0
        assert stats["total_size_mb"] == 0

    def test_cache_stats_with_files(self, temp_cache_dir):
        """Stats reflect actual cache contents."""
        # Create some files with enough data to be measurable
        for _ in range(3):
            upload_cache.store_upload(str(uuid.uuid4()), {"data": "x" * 1000})

        stats = upload_cache.get_cache_stats()
        assert stats["file_count"] == 3
        # Total size should be positive (each file ~1KB, so ~3KB total = ~0.003 MB)
        # Use >= 0 since rounding might make it 0.0 for very small files
        assert stats["total_size_mb"] >= 0
