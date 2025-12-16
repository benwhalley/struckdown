"""
Disk-based cache for batch tasks in the playground.

Stores task state as JSON files on disk, enabling:
- Multi-worker deployments (workers share filesystem)
- Thread-safe atomic updates
- Automatic cleanup of old tasks

Configuration via environment variables:
- STRUCKDOWN_TASK_CACHE_DIR: Cache directory (default: ~/.struckdown/tasks)
- STRUCKDOWN_TASK_MAX_AGE: Max task age in seconds (default: 86400 = 1 day)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import filelock
import validators

logger = logging.getLogger(__name__)

# Configuration
_default_cache_dir = Path("~/.struckdown/tasks").expanduser()
TASK_CACHE_DIR = Path(
    os.environ.get("STRUCKDOWN_TASK_CACHE_DIR", str(_default_cache_dir))
)
TASK_MAX_AGE_SECONDS = int(os.environ.get("STRUCKDOWN_TASK_MAX_AGE", "86400"))  # 1 day


def _ensure_cache_dir() -> None:
    """Ensure the cache directory exists."""
    TASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _task_path(task_id: str) -> Path:
    """Get path to task file."""
    return TASK_CACHE_DIR / f"{task_id}.json"


def _lock_path(task_id: str) -> Path:
    """Get path to task lock file."""
    return TASK_CACHE_DIR / f"{task_id}.lock"


def validate_task_id(task_id: str) -> bool:
    """Validate task_id is a proper UUID."""
    return validators.uuid(task_id) is True


def create_task(task_id: str, data: dict) -> None:
    """Create a new task.

    Args:
        task_id: UUID string identifying the task
        data: Initial task data

    Raises:
        ValueError: If task_id is not a valid UUID
    """
    if not validate_task_id(task_id):
        raise ValueError(f"Invalid task_id: {task_id}")

    _ensure_cache_dir()

    task_data = {
        **data,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    path = _task_path(task_id)
    lock = filelock.FileLock(_lock_path(task_id), timeout=10)

    with lock:
        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(task_data))
        tmp_path.rename(path)

    logger.debug(f"Created task {task_id}")


def get_task(task_id: str) -> Optional[dict]:
    """Retrieve task data.

    Args:
        task_id: UUID string identifying the task

    Returns:
        Task data dict, or None if not found/expired
    """
    if not validate_task_id(task_id):
        return None

    path = _task_path(task_id)
    if not path.exists():
        return None

    lock = filelock.FileLock(_lock_path(task_id), timeout=10)

    try:
        with lock:
            data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, filelock.Timeout):
        return None

    # Check expiry
    if time.time() - data.get("created_at", 0) > TASK_MAX_AGE_SECONDS:
        delete_task(task_id)
        return None

    return data


def update_task(task_id: str, **updates) -> bool:
    """Update specific fields in a task.

    Args:
        task_id: UUID string identifying the task
        **updates: Fields to update

    Returns:
        True if updated successfully, False if task not found
    """
    if not validate_task_id(task_id):
        return False

    path = _task_path(task_id)
    lock = filelock.FileLock(_lock_path(task_id), timeout=10)

    try:
        with lock:
            if not path.exists():
                return False

            data = json.loads(path.read_text())
            data.update(updates)
            data["updated_at"] = time.time()

            # Atomic write
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data))
            tmp_path.rename(path)

        return True
    except (json.JSONDecodeError, OSError, filelock.Timeout):
        return False


def append_result(task_id: str, result: dict) -> bool:
    """Append a result to the task's results list.

    Thread-safe append operation.

    Args:
        task_id: UUID string identifying the task
        result: Result dict to append

    Returns:
        True if appended successfully
    """
    if not validate_task_id(task_id):
        return False

    path = _task_path(task_id)
    lock = filelock.FileLock(_lock_path(task_id), timeout=10)

    try:
        with lock:
            if not path.exists():
                return False

            data = json.loads(path.read_text())
            data.setdefault("results", []).append(result)
            data["completed"] = len(data["results"])
            data["updated_at"] = time.time()

            # Atomic write
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data))
            tmp_path.rename(path)

        return True
    except (json.JSONDecodeError, OSError, filelock.Timeout):
        return False


def append_event(task_id: str, event: dict) -> bool:
    """Append an event to the task's events list.

    Thread-safe append operation.

    Args:
        task_id: UUID string identifying the task
        event: Event dict to append

    Returns:
        True if appended successfully
    """
    if not validate_task_id(task_id):
        return False

    path = _task_path(task_id)
    lock = filelock.FileLock(_lock_path(task_id), timeout=10)

    try:
        with lock:
            if not path.exists():
                return False

            data = json.loads(path.read_text())
            data.setdefault("events", []).append(event)
            data["updated_at"] = time.time()

            # Atomic write
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data))
            tmp_path.rename(path)

        return True
    except (json.JSONDecodeError, OSError, filelock.Timeout):
        return False


def get_new_results(task_id: str, after_index: int) -> tuple[list[dict], int]:
    """Get results added after a given index.

    Args:
        task_id: UUID string identifying the task
        after_index: Return results after this index

    Returns:
        Tuple of (new_results, current_total_count)
    """
    task = get_task(task_id)
    if not task:
        return [], 0

    results = task.get("results", [])
    return results[after_index:], len(results)


def get_new_events(task_id: str, after_index: int) -> tuple[list[dict], int]:
    """Get events added after a given index.

    Args:
        task_id: UUID string identifying the task
        after_index: Return events after this index

    Returns:
        Tuple of (new_events, current_total_count)
    """
    task = get_task(task_id)
    if not task:
        return [], 0

    events = task.get("events", [])
    return events[after_index:], len(events)


def delete_task(task_id: str) -> bool:
    """Delete a task.

    Args:
        task_id: UUID string identifying the task

    Returns:
        True if deleted, False if not found
    """
    if not validate_task_id(task_id):
        return False

    path = _task_path(task_id)
    lock_path = _lock_path(task_id)

    deleted = False
    if path.exists():
        path.unlink(missing_ok=True)
        deleted = True

    lock_path.unlink(missing_ok=True)

    if deleted:
        logger.debug(f"Deleted task {task_id}")

    return deleted


def cleanup_tasks() -> int:
    """Remove expired tasks.

    Returns:
        Number of tasks removed
    """
    if not TASK_CACHE_DIR.exists():
        return 0

    now = time.time()
    removed = 0

    for path in TASK_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            created_at = data.get("created_at", 0)

            if now - created_at > TASK_MAX_AGE_SECONDS:
                task_id = path.stem
                delete_task(task_id)
                removed += 1
                logger.debug(f"Removed expired task: {task_id}")
        except (json.JSONDecodeError, OSError):
            # Remove corrupted files
            path.unlink(missing_ok=True)
            removed += 1

    if removed > 0:
        logger.info(f"Task cleanup: removed {removed} tasks")

    return removed
