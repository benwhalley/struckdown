"""Tests for task_cache module."""

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest import mock

import pytest

from struckdown.playground import task_cache


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "tasks"
    with mock.patch.object(task_cache, "TASK_CACHE_DIR", cache_dir):
        yield cache_dir


class TestValidateTaskId:
    def test_valid_uuid(self):
        task_id = str(uuid.uuid4())
        assert task_cache.validate_task_id(task_id) is True

    def test_invalid_uuid(self):
        assert task_cache.validate_task_id("not-a-uuid") is False
        assert task_cache.validate_task_id("") is False
        assert task_cache.validate_task_id("12345") is False


class TestCreateTask:
    def test_create_and_get(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(
            task_id,
            {
                "status": "pending",
                "total": 10,
                "results": [],
            },
        )

        task = task_cache.get_task(task_id)
        assert task is not None
        assert task["status"] == "pending"
        assert task["total"] == 10
        assert "created_at" in task
        assert "updated_at" in task

    def test_create_invalid_uuid_raises(self, temp_cache_dir):
        with pytest.raises(ValueError):
            task_cache.create_task("invalid", {"status": "pending"})

    def test_atomic_write(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "pending"})

        # Check no .tmp files left behind
        tmp_files = list(temp_cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestGetTask:
    def test_get_nonexistent_returns_none(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        assert task_cache.get_task(task_id) is None

    def test_get_invalid_uuid_returns_none(self, temp_cache_dir):
        assert task_cache.get_task("invalid") is None

    def test_expired_task_returns_none(self, temp_cache_dir):
        task_id = str(uuid.uuid4())

        # Create task with old timestamp
        with mock.patch.object(task_cache, "TASK_MAX_AGE_SECONDS", 1):
            task_cache.create_task(task_id, {"status": "pending"})
            time.sleep(1.1)
            assert task_cache.get_task(task_id) is None


class TestUpdateTask:
    def test_update_existing(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "pending"})

        result = task_cache.update_task(task_id, status="running", extra="data")
        assert result is True

        task = task_cache.get_task(task_id)
        assert task["status"] == "running"
        assert task["extra"] == "data"

    def test_update_nonexistent_returns_false(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        result = task_cache.update_task(task_id, status="running")
        assert result is False


class TestAppendResult:
    def test_append_result(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "running", "results": []})

        task_cache.append_result(task_id, {"index": 0, "outputs": {"a": 1}})
        task_cache.append_result(task_id, {"index": 1, "outputs": {"a": 2}})

        task = task_cache.get_task(task_id)
        assert len(task["results"]) == 2
        assert task["completed"] == 2
        assert task["results"][0]["index"] == 0
        assert task["results"][1]["index"] == 1


class TestAppendEvent:
    def test_append_event(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "running", "events": []})

        task_cache.append_event(task_id, {"type": "slot", "row_index": 0})
        task_cache.append_event(task_id, {"type": "slot", "row_index": 1})

        task = task_cache.get_task(task_id)
        assert len(task["events"]) == 2


class TestGetNewResults:
    def test_get_new_results(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "running", "results": []})

        task_cache.append_result(task_id, {"index": 0})
        task_cache.append_result(task_id, {"index": 1})
        task_cache.append_result(task_id, {"index": 2})

        new_results, total = task_cache.get_new_results(task_id, 1)
        assert len(new_results) == 2
        assert total == 3
        assert new_results[0]["index"] == 1
        assert new_results[1]["index"] == 2


class TestDeleteTask:
    def test_delete_existing(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        task_cache.create_task(task_id, {"status": "pending"})

        result = task_cache.delete_task(task_id)
        assert result is True
        assert task_cache.get_task(task_id) is None

    def test_delete_nonexistent(self, temp_cache_dir):
        task_id = str(uuid.uuid4())
        result = task_cache.delete_task(task_id)
        assert result is False


class TestCleanupTasks:
    def test_cleanup_expired(self, temp_cache_dir):
        with mock.patch.object(task_cache, "TASK_MAX_AGE_SECONDS", 1):
            task_id = str(uuid.uuid4())
            task_cache.create_task(task_id, {"status": "complete"})

            time.sleep(1.1)
            removed = task_cache.cleanup_tasks()
            assert removed == 1
            assert task_cache.get_task(task_id) is None
